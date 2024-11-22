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
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from scipy.ndimage import gaussian_filter1d
import argparse
import joblib 

label_encoder = LabelEncoder()
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process MEG data for a specific subject.')
parser.add_argument('--subject', type=str, required=True, help='Subject identifier')
args = parser.parse_args()

# Use the subject from the command-line argument
subj = "R2490"

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

window_size = 100  # in milliseconds
step_size = 50     # in milliseconds
sampling_rate = raw.info['sfreq']  # Hz
window_samples = int(window_size * sampling_rate / 1000)
step_samples = int(step_size * sampling_rate / 1000)
labels_df = pd.read_csv(f'{label_dir}/{subj}/label.csv')


label_dict = dict(zip(labels_df['trial_index'], labels_df['trial.rule']))

# Step 3: Extract labels for the valid trials in trial_info_valid
y_labels = []
for info in trial_info:
    idx = info['trial_index']
    if idx in label_dict:
        y_labels.append(label_dict[idx])

X = np.array([md.data for md in epochs_data_list])  # Shape: (n_epochs, n_channels, n_times)
X = X.squeeze(axis=1)
# Extract labels
y = label_encoder.fit_transform(y_labels)  # Convert labels to integers


# Print the number of labels and valid trials after matching
n_labels = len(y_labels)
n_trials = len(trial_info)
print(f"Number of labels after matching: {n_labels}")
print(f"Number of valid trials after matching: {n_trials}")

# Assuming X is your feature matrix with shape (n_epochs, n_channels, n_times)
X = np.array([md.data for md in epochs_data_list])  # Ensure this is 3D
X = X.squeeze(axis=1)  # This might reduce dimensions, ensure it's still 3D


# Sliding window loop
time_windows = []
accuracies = []

for start in range(0, X.shape[2] - window_samples + 1, step_samples):
    end = start + window_samples
    time_windows.append((start, end))
    
    # Extract features for the current window
    features = X[:, :, start:end].reshape(X.shape[0], -1)  # Flatten Channels x Time
    cv = LeaveOneOut()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    
    accuracy = cross_val_score(clf, features, y, cv=cv).mean()
    accuracies.append(accuracy)

# Plot decoding accuracy over time
import matplotlib.pyplot as plt

time_points = [np.mean(window) / sampling_rate * 1000 for window in time_windows]  # Convert to ms
plt.plot(time_points, accuracies)
plt.xlabel('Time (ms)')
plt.ylabel('Decoding Accuracy')
plt.title('Sliding Window Decoding Accuracy')
plt.show()
plt.savefig(f'{save_dir}/{subj}/decoding_accuracy.png')