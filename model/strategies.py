import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np 

"""
NOTE: Observations are coded as 2D matrices with values -1 for un-revealed, 0 for revealed white, and 1 revealed red. 
"""
class MazeCNN(nn.Module):
    def __init__(self, embedding_dim=32):
        super(MazeCNN, self).__init__()
        # Simple CNN architecture
        # Input: 1 channel (observation with -1, 0, 1) -> Encode to 3 channels (one-hot)
        # then process through convolutional layers
        
        self.embedding_dim = embedding_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, embedding_dim, kernel_size=3, padding=1)
        
        # Decoder layers (to reconstruct the full maze)
        self.deconv1 = nn.Conv2d(embedding_dim, 32, kernel_size=3, padding=1)
        self.deconv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.deconv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)  # Output: probabilities for each tile
        
    def encode(self, x):
        """
        Encode an observation into an embedding
        x: batch of observations [B, 7, 7] with values -1, 0, 1
        """
        # Convert to one-hot encoding
        x_onehot = self._to_onehot(x)  # [B, 3, 7, 7]
        
        # Forward pass through encoder
        x = F.relu(self.conv1(x_onehot))  # [B, 16, 7, 7]
        x = F.relu(self.conv2(x))  # [B, 32, 7, 7]
        embedding = self.conv3(x)  # [B, embedding_dim, 7, 7]
        
        return embedding
        
    def decode(self, embedding):
        """
        Decode an embedding into a full maze prediction
        """
        x = F.relu(self.deconv1(embedding))  # [B, 32, 7, 7]
        x = F.relu(self.deconv2(x))  # [B, 16, 7, 7]
        logits = self.deconv3(x)  # [B, 1, 7, 7]
        
        return logits.squeeze(1)  # [B, 7, 7]
    
    def forward(self, x):
        """
        Forward pass through encoder and decoder
        x: batch of observations [B, 7, 7] with values -1, 0, 1
        """
        embedding = self.encode(x)
        logits = self.decode(embedding)
        return logits
    
    def _to_onehot(self, x):
        """
        Convert observations with values -1, 0, 1 to one-hot encoding
        x: [B, 7, 7]
        returns: [B, 3, 7, 7]
        """
        batch_size = x.size(0)
        # Create one-hot tensors
        x_onehot = torch.zeros(batch_size, 3, 7, 7, device=x.device)
        
        # Set the appropriate channels
        x_onehot[:, 0, :, :] = (x == -1)  # Channel 0: covered tiles (-1)
        x_onehot[:, 1, :, :] = (x == 0)   # Channel 1: revealed 0s
        x_onehot[:, 2, :, :] = (x == 1)   # Channel 2: revealed 1s
        
        return x_onehot.float()
    
    def compute_similarity(self, obs1, obs2):
        """
        Compute similarity between two observations using their embeddings
        obs1, obs2: observations [7, 7] with values -1, 0, 1
        """
        # Add batch dimension if necessary
        if obs1.dim() == 2:
            obs1 = obs1.unsqueeze(0)
        if obs2.dim() == 2:
            obs2 = obs2.unsqueeze(0)
        
        # Compute embeddings
        emb1 = self.encode(obs1)  # [1, embedding_dim, 7, 7]
        emb2 = self.encode(obs2)  # [1, embedding_dim, 7, 7]
        
        # Flatten embeddings
        emb1_flat = emb1.view(1, -1)  # [1, embedding_dim * 7 * 7]
        emb2_flat = emb2.view(1, -1)  # [1, embedding_dim * 7 * 7]
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(emb1_flat, emb2_flat)
        
        return similarity.item()
    

class MazeEpisodicMemoryModel(nn.Module):
    def __init__(self, grid_size=7, memory_size=10, temperature=1.0,use_similarity=True):
        """
        Initialize the episodic memory model for maze game.
        
        Args:
            grid_size: Size of the maze grid (grid_size x grid_size)
            memory_size: Number of previous mazes to remember (N)
            temperature: Value for softmax temperature parameter
            use_similarity: Whether to use a similarity function to weight previous trial boards. 
        """
        super(MazeEpisodicMemoryModel, self).__init__()
        
        self.grid_size = grid_size
        self.memory_size = memory_size
        self.use_similarity=use_similarity
        
        # Memory structure:
        # For each maze, we store:
        # 1. The complete maze
        # 2. A list of all observations seen for that maze
        self.memory = deque(maxlen=memory_size)  # Using deque for O(1) popping
        
        # Cache for embeddings to avoid recomputing
        self.embedding_dict = {}
        
        # Load pre-trained CNN
        self.cnn = MazeCNN(embedding_dim=16)
        self.cnn.load_state_dict(torch.load('data/mazecnn_weights.pt'))
        self.cnn.cpu()
        self.cnn.eval()  # Set to evaluation mode
        
        # Fixed temperature parameter
        self.temperature = temperature
        
    def get_embedding(self, obs):
        """
        Get embedding for an observation, using cache if available.
        
        Args:
            obs: Observation tensor [grid_size, grid_size]
            
        Returns:
            embedding: Tensor of shape [embedding_dim * grid_size * grid_size]
        """
        obs_key = tuple(obs.flatten().tolist())
        
        if obs_key in self.embedding_dict:
            return self.embedding_dict[obs_key]
        
        with torch.no_grad():  # Disable gradient calculation for inference
            emb = self.cnn.encode(obs.reshape(1, self.grid_size, self.grid_size))
            self.embedding_dict[obs_key] = emb
        
        return emb
    
    def batch_check_consistency(self, observation, mazes):
        """
        Fully vectorized consistency check for multiple mazes at once.
        
        Args:
            observation: Current observation tensor [grid_size, grid_size]
            mazes: Tensor of mazes [num_mazes, grid_size, grid_size]
            
        Returns:
            mask: Boolean tensor indicating which mazes are consistent [num_mazes]
        """
        # Get revealed positions mask
        revealed_mask = (observation != -1)
        
        if not torch.any(revealed_mask):
            # If no revealed tiles, all mazes are consistent
            return torch.ones(len(mazes), dtype=torch.bool)
        
        # Flatten everything for easier operations
        obs_flat = observation.flatten()  # [49]
        mazes_flat = mazes.reshape(mazes.size(0), -1)  # [num_mazes, 49]
        revealed_mask_flat = revealed_mask.flatten()  # [49]
        
        # Extract only the revealed positions for comparison
        revealed_positions = torch.nonzero(revealed_mask_flat, as_tuple=True)[0]  # indices where tiles are revealed
        
        # Extract values at revealed positions
        revealed_obs_values = obs_flat[revealed_positions]  # [num_revealed]
        revealed_maze_values = mazes_flat[:, revealed_positions]  # [num_mazes, num_revealed]
        
        # Check equality across all revealed positions at once
        # For each maze, we get a boolean tensor of shape [num_revealed]
        equality_checks = (revealed_maze_values == revealed_obs_values.unsqueeze(0))
        
        # A maze is consistent if ALL revealed positions match
        consistent_mask = torch.all(equality_checks, dim=1)  # [num_mazes]
        
        return consistent_mask
        
    def compute_likelihood(self, observation):
        """
        Fully vectorized computation of likelihood based on episodic memory.
        
        Args:
            observation: Current game observation [grid_size, grid_size]
            
        Returns:
            likelihood: Likelihood map for each position [grid_size, grid_size]
        """
        # Convert observation to tensor if not already
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        
        # Initialize likelihood map with zeros
        likelihood = torch.zeros((self.grid_size, self.grid_size), dtype=torch.float32)
        
        # Identify covered indices once
        covered_indices = (observation == -1)
        
        # If no memory yet or no covered tiles, use uniform likelihood for covered tiles
        if len(self.memory) == 0 or not torch.any(covered_indices):
            if torch.any(covered_indices):
                likelihood[covered_indices] = 1.0 / torch.sum(covered_indices).item()
            return likelihood
        
        # Get all mazes and check which ones are consistent
        all_mazes = [m[0] for m in self.memory]
        mazes_tensor = torch.stack(all_mazes) if all_mazes else torch.tensor([])
        
        if len(mazes_tensor) == 0:
            # No mazes in memory
            if torch.any(covered_indices):
                likelihood[covered_indices] = 1.0 / torch.sum(covered_indices).item()
            return likelihood
        
        # Get consistent maze indices using vectorized check
        consistent_mask = self.batch_check_consistency(observation, mazes_tensor)
        consistent_indices = torch.nonzero(consistent_mask, as_tuple=True)[0]
        
        if len(consistent_indices) == 0:
            # No consistent mazes found
            if torch.any(covered_indices):
                likelihood[covered_indices] = 1.0 / torch.sum(covered_indices).item()
            return likelihood

        if self.use_similarity: #Weigh previous mazes by previous observations' visual similarities. 
        
            # Get embedding for current observation
            current_emb = self.get_embedding(observation)
            current_emb_flat = current_emb.view(-1)
            
            # Collect all consistent observations and their maze indices
            all_consistent_obs = []
            maze_indices = []
            
            # Only collect observations from consistent mazes
            for idx in consistent_indices:
                maze_idx = idx.item()
                maze, observations = self.memory[maze_idx]
                
                for stored_obs in observations:
                    all_consistent_obs.append(stored_obs)
                    maze_indices.append(maze_idx)
            
            # If no observations found
            if not all_consistent_obs:
                if torch.any(covered_indices):
                    likelihood[covered_indices] = 1.0 / torch.sum(covered_indices).item()
                return likelihood
            
            # Get embeddings for all observations at once
            all_obs_embeddings = []
            for obs in all_consistent_obs:
                all_obs_embeddings.append(self.get_embedding(obs).view(-1))
            
            # Stack embeddings into a single tensor
            all_embs_tensor = torch.stack(all_obs_embeddings)  # [num_obs, embedding_dim*7*7]
            
            # Compute all cosine similarities at once
            # Normalize current embedding
            current_emb_norm = F.normalize(current_emb_flat.unsqueeze(0), p=2, dim=1)
            
            # Normalize all observation embeddings
            all_embs_norm = F.normalize(all_embs_tensor, p=2, dim=1)
            
            # Compute dot product for all pairs at once
            similarities = torch.mm(current_emb_norm, all_embs_norm.t()).squeeze()
            
            # Apply softmax with temperature to similarities
            weights = F.softmax(similarities / self.temperature, dim=0)
            
            # Calculate weighted sum for each maze
            maze_weight_sum = {}
            
            # Aggregate weights by maze
            for i, maze_idx in enumerate(maze_indices):
                if maze_idx not in maze_weight_sum:
                    maze_weight_sum[maze_idx] = 0.0
                maze_weight_sum[maze_idx] += weights[i].item()
            
            # Apply weighted sum to get likelihood
            for maze_idx, weight in maze_weight_sum.items():
                maze = self.memory[maze_idx][0]
                likelihood[covered_indices] += weight * maze[covered_indices]
            
            likelihood=likelihood/torch.sum(likelihood)
        else:
            for idx in consistent_indices:
                likelihood[covered_indices]+=maze[covered_indices]
            likelihood=likelihood/torch.sum(likelihood)

        return likelihood
    
    def update_memory(self, maze, observation):
        """
        Update episodic memory with a new maze and its observation.
        
        Args:
            maze: Complete maze array [grid_size, grid_size]
            observation: Current observation [grid_size, grid_size]
        """
        # Convert to tensors if not already
        if not isinstance(maze, torch.Tensor):
            maze = torch.tensor(maze, dtype=torch.float32)
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        
        # Check if this maze is already in memory
        for i, (stored_maze, stored_observations) in enumerate(self.memory):
            if torch.all(stored_maze == maze):
                # Maze already exists, add this observation to its list
                self.memory[i][1].append(observation.clone())
                return
        
        # This is a new maze, add it with its first observation
        self.memory.append((maze.clone(), [observation.clone()]))
    
    def batch_update_memory(self, maze, observations):
        """
        Update memory with multiple observations for a maze at once.
        
        Args:
            maze: Complete maze array [grid_size, grid_size]
            observations: List of observations for this maze
        """
        # Convert to tensors if not already
        if not isinstance(maze, torch.Tensor):
            maze = torch.tensor(maze, dtype=torch.float32)
        
        # Convert all observations to tensors
        obs_tensors = []
        for obs in observations:
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32)
            obs_tensors.append(obs.clone())
        
        # Check if this maze is already in memory
        for i, (stored_maze, stored_observations) in enumerate(self.memory):
            if torch.all(stored_maze == maze):
                # Maze already exists, add these observations to its list
                self.memory[i][1].extend(obs_tensors)
                return
        
        # This is a new maze, add it with all its observations
        self.memory.append((maze.clone(), obs_tensors))

    def log_likelihood(self, observations, actions, maze):
        """
        Compute the negative log-likelihood of the observed actions given the observations.
        
        Args:
            observations: List of observations [num_steps, grid_size, grid_size]
            actions: List of (i,j) action coordinates taken [num_steps, 2]
            maze: The complete maze for this episode
            
        Returns:
            neg_log_likelihood: The negative log-likelihood of the observed actions
        """
        neg_log_likelihood = 0.0
        
        for t in range(len(observations)):
            obs = observations[t]
            action = actions[t]
            i, j = action
            
            # Compute likelihood map
            likelihood_map = self.compute_likelihood(obs)
            
            # Get likelihood of the chosen action
            action_likelihood = likelihood_map[i, j].item()
            
            # Add to negative log-likelihood (avoid log(0))
            if action_likelihood > 0:
                neg_log_likelihood -= torch.log(torch.tensor(action_likelihood))
            else:
                neg_log_likelihood += 100.0  # Large penalty for zero likelihood actions
            
            # Skip verification for speed
        
        return neg_log_likelihood

def get_model_likelihood(model, data, batch_update=True):
    """
    Calculate the likelihood of the model on the given data.
    
    Args:
        model: The MazeEpisodicMemoryModel
        data: List of (maze, observations, actions) tuples
        batch_update: Whether to update memory in batches
        
    Returns:
        neg_log_likelihood: The negative log-likelihood of the data
    """
    neg_log_likelihood = 0.0
    
    for maze, observations, actions in data:
        # Calculate log likelihood
        neg_log_likelihood += model.log_likelihood(observations, actions, maze)
        
        # Update memory - more efficient to do in batch
        if batch_update:
            model.batch_update_memory(maze, observations)
        else:
            for obs in observations:
                model.update_memory(maze, obs)
    
    return neg_log_likelihood


def consistent(o,b):
    if (b[np.where(o==0)]!=0).sum()>0:
        return False 
    elif (b[np.where(o==1)]!=1).sum()>0:
        return False 
    else:
        return True 

all_boards_dict=np.load('all_grammar_boards.npz')
all_boards=np.concatenate([all_boards_dict[k] for k in all_boards_dict.keys()])

def grammar_strategy(obs):
    """
    The semantic strategy policy. Takes in a single observation (doesnt depend on any specific trial history)
    """
    possible_chains=np.asarray([b for b in all_boards_dict['chain'] if consistent(obs,b)])
    possible_trees=np.asarray([b for b in all_boards_dict['tree'] if consistent(obs,b)])
    possible_loop=np.asarray([b for b in all_boards_dict['loop'] if consistent(obs,b)])
    rule_possibilities=[possible_chains,possible_trees,possible_loop]
    rule_probs=[len(possible_chains)/len(all_boards_dict['chain']),len(possible_trees)/len(all_boards_dict['tree']),len(possible_loop)/len(all_boards_dict['loop'])]
    possible_boards=rule_possibilities[np.argmax(rule_probs)]
    #print(possible_boards)
    policy=possible_boards.sum(axis=0)
    policy[obs==0]=0
    policy[obs==1]=0
    return policy/np.sum(policy)

def KL_Divergence(P,Q):
     """
     Measure the KL Divergence of two likelihood maps (e.g. semantic and episodic)
     """
     epsilon = 0.00001

     # You may want to instead make copies to avoid changing the np arrays.
     P = P+epsilon
     Q = Q+epsilon

     divergence = np.sum(P*np.log(P/Q))
     return divergence