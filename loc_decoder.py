import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer, GeneralizingEstimator
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegressionCV
from scipy.ndimage import gaussian_filter1d
import argparse
import joblib 

label_encoder = LabelEncoder()
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process MEG data for a specific subject.')
parser.add_argument('--subject', type=str, required=True, help='Subject identifier')
args = parser.parse_args()

# Use the subject from the command-line argument
subj = args.subject

data_dir = 'data_meg'
dataqual = 'prepro' #or loc/exp
exp = 'exp' #or exp
dtype = "raw"
label_dir = 'data_log'
save_dir = 'data_meg'
raw = mne.io.read_raw_fif(f'{data_dir}/{subj}/{dataqual}/{subj}_{exp}.fif', preload=True)
bad_channels_dict = {
    "R2490": ['MEG 014', 'MEG 004', 'MEG 079', 'MEG 072', 'MEG 070', 'MEG 080', 'MEG 074', 'MEG 067', 'MEG 082', 'MEG 105', 'MEG 115', 'MEG 141', 'MEG 153'],
    "R2488": ['MEG 015', 'MEG 014', 'MEG 068', 'MEG 079', 'MEG 146', 'MEG 147', 'MEG 007', 'MEG 141'],
    "R2487": ['MEG 015', 'MEG 014', 'MEG 068', 'MEG 079', 'MEG 147', 'MEG 146', 'MEG 004'],
    "R2280": ['MEG 015', 'MEG 039', 'MEG 077', 'MEG 076', 'MEG 073', 'MEG 079', 'MEG 064', 'MEG 059', 'MEG 070']
}
bad_channels = bad_channels_dict.get(subj, [])
raw.info['bads'].extend(bad_channels)
raw.drop_channels(bad_channels)
# Initialize lists to store individual epochs data and trial information
loclizers = [4, 9, 16, 25]
trail_t = 21
tmin = -0.2  
epochs_data_list = []
trial_info_valid = []
sfreq = raw.info['sfreq']
raw.filter(1, 40, method='iir')
downsample = 10
raw.resample(sfreq / downsample)


events = mne.find_events(raw, stim_channel='STI 014', output='onset', shortest_event=1)
event_id = {
    'start': 160,
    'move': 161,
    'reveal_red': 162,
    'reveal_white': 163,
    'done': 164,
    'choice': 165,
    'timeout': 166
}

# Initialize a list to store trial information
trial_info = []
previous_start_sample = None
processed_starts = set()
start_idx = 0
start_events = events[events[:, 2] == event_id['start']]
done_events = events[events[:, 2] == event_id['done']]
timeout_events = events[events[:, 2] == event_id['timeout']]
reveal_red_events = events[events[:, 2] == event_id['reveal_red']]
reveal_white_events = events[events[:, 2] == event_id['reveal_white']]
sfreq = raw.info['sfreq']  # Sampling frequency

# Combine 'done' and 'timeout' events
end_events = np.concatenate((done_events, timeout_events))
end_events = end_events[end_events[:, 0].argsort()]  # Sort by time


# Combine 'done' and 'timeout' events
end_events = np.concatenate((done_events, timeout_events))
end_events = end_events[end_events[:, 0].argsort()]  # Sort by time

# Iterate through each start event to create trial information
for idx, start_event in enumerate(start_events):
    start_sample = start_event[0]
    
    # Find the next end event after the start event
    end_idx = np.searchsorted(end_events[:, 0], start_sample, side='right')
    end_sample = None
    while end_idx < len(end_events):
        potential_end_sample = end_events[end_idx, 0]
        if (potential_end_sample - start_sample) / sfreq <= trail_t:
            end_sample = potential_end_sample
            break
        end_idx += 1

    # If no valid end event is found, set end_sample to 26 seconds after start_sample
    if end_sample is None:
        end_sample = start_sample + int(trail_t * sfreq)
        

    # Check if the current start is at least 20 seconds after the previous start
    if (previous_start_sample is None or (start_sample - previous_start_sample) / sfreq >= 20) and start_sample not in processed_starts:
        # Proceed with processing this start event
        previous_start_sample = start_sample
        processed_starts.add(start_sample)  # Add to the set of processed starts

        # Calculate tmin and tmax for the epoch
        tmin = -0.2  # 0.2 s before 'start'
        tmax = trail_t  # Duration from 'start' to 1 s after end event
        reveal_red_within_trial = reveal_red_events[(reveal_red_events[:, 0] > start_sample) & 
                                                    (reveal_red_events[:, 0] < end_sample)]
        reveal_white_within_trial = reveal_white_events[(reveal_white_events[:, 0] > start_sample) & 
                                                        (reveal_white_events[:, 0] < end_sample)]
        # Store trial information, including whether 'reveal_red' and 'reveal_white' occurred
        trial_info.append({
            'event_sample': start_sample,
            'trial_index': start_idx,
            'duration': tmax,
            'tmin': tmin,
            'tmax': trail_t,
            'done': len(done_events) > 0,
            'start_times': start_sample / sfreq,
            'done_times': end_sample / sfreq,
            'reveal_red': len(reveal_red_within_trial) > 0,  # Boolean flag indicating if 'reveal_red' occurred
            'reveal_red_times': (reveal_red_within_trial[:, 0] - start_sample) / sfreq if len(reveal_red_within_trial) > 0 else [],
            'reveal_white': len(reveal_white_within_trial) > 0,  # Boolean flag indicating if 'reveal_white' occurred
            'reveal_white_times': (reveal_white_within_trial[:, 0] - start_sample) / sfreq if len(reveal_white_within_trial) > 0 else [],
            'reveal_times': sorted(
                ((reveal_red_within_trial[:, 0] - start_sample) / sfreq).tolist() +
                ((reveal_white_within_trial[:, 0] - start_sample) / sfreq).tolist()
            )
        })
        start_idx += 1
new_events = np.array([[info['event_sample'], 0, event_id['start']] for info in trial_info])

# Iterate over new_events and create epochs, skipping the unwanted trials
for idx, event in enumerate(new_events):
    start_sample = event[0]
    event_id_code = event[2]
    event_time = start_sample / sfreq
    total_duration = raw.times[-1]
    print(f"Tmax is {tmax}")
    picks = mne.pick_types(raw.info, meg=True, exclude='bads')

    # Create the epoch
    epochs = mne.Epochs(
        raw, [event], event_id={f'event_{event_id_code}': event_id_code},
        tmin=tmin, tmax=26, preload=True,picks=picks,
        reject_by_annotation=False, reject=None, verbose=True
    )

    # Append valid epochs to the list
    if len(epochs) > 0:
        epochs_data_list.append(epochs.get_data()[:, :, ::downsample])
        trial_info_valid.append(trial_info[idx])
    else:
        print(f"Epoch {idx} was dropped.")
        print(f"Drop log for Epoch {idx}: {epochs.drop_log}")


# Step 0: Load labels
label_encoder = LabelEncoder()
labels_df = pd.read_csv(f'{label_dir}/{subj}/label.csv')

# Step 1: Get the valid trial indices
valid_trial_indices = {info['trial_index'] for info in trial_info_valid}

# Step 2: Filter labels_df to only include valid trial indices
labels_df_filtered = labels_df[labels_df['trial_index'].isin(valid_trial_indices)]
label_dict = dict(zip(labels_df_filtered['trial_index'], labels_df_filtered['trial.rule']))

X = np.array([md.data for md in epochs_data_list])  # Shape: (n_epochs, n_channels, n_times)
X = X.squeeze(axis=1)
# Step 3: Extract labels for the valid trials in trial_info_valid
y_labels = []
for info in trial_info_valid:
    idx = info['trial_index']
    if idx in label_dict:
        y_labels.append(label_dict[idx])

# Convert labels to integers using label encoder
y = label_encoder.fit_transform(y_labels)

# Print the number of labels and valid trials after matching
n_labels = len(y_labels)
n_trials = len(trial_info_valid)

def extract_reveal(trial_info_valid, raw, label_dict, event_name='reveal_red', n_points_before=50, n_points_after=50, num_events=4):
    y_labels = []
    X_reveal = []
    trial_indices = []
    sfreq = raw.info['sfreq']  # Sampling frequency

    for info in trial_info_valid:
        reveal_times = info[f'{event_name}_times']
        if len(reveal_times) >= num_events:
            trial_data = []

            for event_time in reveal_times[:num_events]:
                event_sample = int(event_time * sfreq)
                start_sample_before = event_sample - n_points_before
                end_sample_after = event_sample + n_points_after

                if start_sample_before >= 0 and end_sample_after <= raw.n_times:
                    epoch_data = raw.get_data(start=start_sample_before, stop=end_sample_after)
                    
                    # Ensure epoch_data is 3D
                    if epoch_data.ndim == 2:
                        epoch_data = np.expand_dims(epoch_data, axis=2)
                    
                    trial_data.append(epoch_data)

            if trial_data:
                # Check if all trial_data have the same shape
                trial_data_shapes = [data.shape for data in trial_data]
                if len(set(trial_data_shapes)) == 1:
                    trial_data_concatenated = np.concatenate(trial_data, axis=2)
                    X_reveal.append(trial_data_concatenated)
                    trial_index = info['trial_index']
                    if trial_index in label_dict:
                        y_labels.append(label_dict[trial_index])
                        trial_indices.append(trial_index)
                else:
                    print(f"Inconsistent shapes in trial data for trial index {info['trial_index']}: {trial_data_shapes}")

    return np.array(X_reveal), np.array(y_labels), trial_indices

def extract_start(trial_info, raw, label_dict, n_points_before=50, n_points_after=50):
    X_start = []
    y_labels = []
    trial_indices = []
    for info in trial_info:
        # Get the done sample from the trial information
        start_sample = info['event_sample'] + int((info['tmin'] + 0.2) * raw.info['sfreq'])

        # Calculate start and end samples for extraction
        start_sample_before = start_sample - n_points_before
        end_sample_after = start_sample + n_points_after

        # Ensure the samples are within bounds
        if start_sample_before >= 0 and end_sample_after <= raw.n_times:
            # Extract data for this trial
            epoch_data = raw.get_data(start=start_sample_before, stop=end_sample_after)
            X_start.append(epoch_data)
            trial_index = info['trial_index']
            if trial_index in label_dict:
                y_labels.append(label_dict[trial_index])
                trial_indices.append(trial_index)

    return  np.array(X_start), np.array(y_labels), trial_indices


def extract_done(trial_info, raw, label_dict, n_points_before=50, n_points_after=50):
    X_done = []
    y_labels = []
    trial_indices = []
    sfreq = raw.info['sfreq']  # Sampling frequency
    done_times = [info['done_times'] for info in trial_info]
    start_times = [info['start_times'] for info in trial_info]
    
    for info, done_time, start_time in zip(trial_info, done_times, start_times):
        # Check if the "done" event exists
        done_time = info['done_times'] 
        start_time = info['start_times']
        print(f"Trial_index: {info['trial_index']}, Done time: {done_time}, Start time: {start_time}")
        if int(done_time) - int(start_time) > 25:  # Assuming these indicate the presence of a "done" event
            done_sample = info['event_sample'] + int((info['tmax'] - 1.0) * sfreq)
            start_sample_before = done_sample - n_points_before
            end_sample_after = done_sample + n_points_after
        else:
            start_sample_before = int(done_time * sfreq) - n_points_before
            end_sample_after = int(done_time * sfreq) + n_points_after

        # Ensure the samples are within bounds
        if start_sample_before >= 0 and end_sample_after <= raw.n_times:
            epoch_data = raw.get_data(start=start_sample_before, stop=end_sample_after)
            X_done.append(epoch_data)
            trial_index = info['trial_index']
            if trial_index in label_dict:
                y_labels.append(label_dict[trial_index])
                trial_indices.append(trial_index)

    return np.array(X_done), np.array(y_labels), trial_indices


# Extract the first n_events reveal events
epoch_reveal, y_labels_reveal, trial_indices_reveal = extract_reveal(trial_info_valid, raw, label_dict,num_events=4)
epoch_start, y_labels_start, trial_indices_start = extract_start(trial_info_valid, raw, label_dict)
epoch_done, y_labels_done, trial_indices_done = extract_done(trial_info_valid, raw, label_dict)

# Find common trial indices across all epochs
common_trial_indices = set(trial_indices_reveal) & set(trial_indices_start) & set(trial_indices_done)

# Convert to a sorted list to maintain order
common_trial_indices = sorted(common_trial_indices)

# Function to filter epochs and labels based on common trial indices
def filter_by_trial_indices(epochs, labels, trial_indices, common_indices):
    mask = [i for i, idx in enumerate(trial_indices) if idx in common_indices]
    return epochs[mask], labels[mask]

# Filter each epoch and labels
epoch_start_filtered, y_labels_start_filtered = filter_by_trial_indices(epoch_start, y_labels_start, trial_indices_start, common_trial_indices)
epoch_reveal_filtered, y_labels_reveal_filtered = filter_by_trial_indices(epoch_reveal, y_labels_reveal, trial_indices_reveal, common_trial_indices)
epoch_done_filtered, y_labels_done_filtered = filter_by_trial_indices(epoch_done, y_labels_done, trial_indices_done, common_trial_indices)

# Flatten the reveal epoch to combine the event dimension into the time axis
n_trials, n_channels, n_timepoints_per_event, n_events = epoch_reveal_filtered.shape
epoch_reveal_flattened = epoch_reveal_filtered.reshape(n_trials, n_channels, n_timepoints_per_event * n_events)

# Check the new shape
print(f"Epoch_reveal shape: {epoch_reveal_filtered.shape}")
print(f"Flattened epoch_reveal shape: {epoch_reveal_flattened.shape}")

# Combine the filtered epochs
X_combined = np.concatenate((epoch_start_filtered, epoch_reveal_flattened, epoch_done_filtered), axis=2)

# Use the labels from the first filtered epoch (assuming they are the same for all)
y_combined = y_labels_done_filtered

n_time_points = X_combined.shape[2]
n_classes = len(np.unique(y_combined))


def smooth_scores(scores, sigma=2):
    return gaussian_filter1d(scores, sigma=sigma)

def segment_data_into_windows(data, sfreq, window_size=1):
    """
    Segments data into windows of specified size.
    
    Parameters:
    - data: 3D numpy array of shape (n_samples, n_channels, n_timepoints)
    - sfreq: Sampling frequency of the data
    - window_size: Size of each window in seconds
    
    Returns:
    - windows: List of 3D numpy arrays, each representing a window
    """
    n_samples, n_channels, n_timepoints = data.shape
    window_length = int(sfreq * window_size)
    windows = []
    
    for start in range(0, n_timepoints - window_length + 1, window_length):
        end = start + window_length
        window = data[:, :, start:end]
        windows.append(window)
    
    return windows

def locolizer(subj, group, X_new):
    # Construct the filename
    model_filename = f"locolizer/{subj}_decoder_{group}.joblib"
    
    # Load the model
    clf = joblib.load(model_filename)
    print(f"Model for group {group} loaded from {model_filename}")
    
    # Ensure the new data is in the correct shape
    # Assuming X_new is 3D: (n_samples, n_channels, n_timepoints)
    X_new_flat = X_new.reshape(X_new.shape[0], -1)  # Flatten the data
    
    # Use the model to make predictions
    predictions = clf.predict(X_new_flat)
    return predictions


sfreq = raw.info['sfreq'] 
accuracies = []
windows = segment_data_into_windows(X_combined, sfreq, window_size=1)
for loc in loclizers:
    all_predictions = []
    for window in windows:
        predictions = locolizer(subj, loc, window)
        all_predictions.extend(predictions)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_combined, all_predictions)
    accuracies.append(accuracy)
    print(f"Accuracy for locolizer {loc}: {accuracy}")

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.plot(loclizers, accuracies, marker='o')
plt.title('Prediction Accuracy for Different Loclizers')
plt.xlabel('Loclizer')
plt.ylabel('Accuracy')
plt.xticks(loclizers)
plt.ylim(0, 1)
plt.grid(True)
plt.show()

