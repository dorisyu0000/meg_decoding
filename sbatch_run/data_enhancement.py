import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import pandas as pd
import mne
from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer, GeneralizingEstimator
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from matplotlib import pyplot as plt
label_encoder = LabelEncoder()
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from scipy.ndimage import gaussian_filter1d
import argparse
from utils import EventExtractor

parser = argparse.ArgumentParser(description='Process MEG data for a specific subject.')
parser.add_argument('--subject', type=str, required=True, help='Subject identifier')
args = parser.parse_args()
subj = args.subject
# subj = 'R2487'
data_dir = 'data_meg'
dataqual = 'prepro' #or loc/exp
exp = 'exp' #or exp
dtype = "raw"
label_dir = 'data_log'
save_dir = 'data_meg'
raw_path = f'{data_dir}/{subj}/{dataqual}/{subj}_{exp}.fif'
bad_channels_dict = {
    "R2490": ['MEG 014', 'MEG 004', 'MEG 079', 'MEG 072', 'MEG 070', 'MEG 080', 'MEG 074', 'MEG 067', 'MEG 082', 'MEG 105', 'MEG 115', 'MEG 141', 'MEG 153'],
    "R2488": ['MEG 015', 'MEG 014', 'MEG 068', 'MEG 079', 'MEG 146', 'MEG 147', 'MEG 007', 'MEG 141'],
    "R2487": ['MEG 015', 'MEG 014', 'MEG 068', 'MEG 079', 'MEG 147', 'MEG 146', 'MEG 004'],
    "R2280": ['MEG 024', 'MEG 039', 'MEG 079', 'MEG 077', 'MEG 141', 'MEG 073', 'MEG 075', 'MEG 076', 'MEG 064', 'MEG 063', 'MEG 060', 'MEG 059', 'MEG 058']
}
bad_channels = bad_channels_dict.get(subj, [])

# Load raw data
raw = mne.io.read_raw_fif(raw_path).load_data()
raw.info['bads'].extend(bad_channels)
reject = dict(mag=5e-12, grad=4000e-13)
raw.filter(1, 30, fir_design="firwin")
sfreq = raw.info['sfreq']
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

# Define trials to remove
trials_to_remove = []
start_events = events[events[:, 2] == event_id['start']]
done_events = events[events[:, 2] == event_id['done']]
timeout_events = events[events[:, 2] == event_id['timeout']]
reveal_red_events = events[events[:, 2] == event_id['reveal_red']]
reveal_white_events = events[events[:, 2] == event_id['reveal_white']]
sfreq = raw.info['sfreq']  # Sampling frequency

# Combine 'done' and 'timeout' events
end_events = np.concatenate((done_events, timeout_events))
end_events = end_events[end_events[:, 0].argsort()] 

filtered_start_events = [start_events[0]]  # Start with the first event

# Check for at least 3 seconds between each done event
for i in range(1, len(start_events)):
    time_diff = (start_events[i, 0] - start_events[i-1, 0]) / sfreq
    if time_diff < 2:
        print(f"Warning: Less than 3 seconds between done events at indices {i-1} and {i}")
    
    else:
        filtered_start_events.append(start_events[i])
        
start_events = filtered_start_events

# Initialize a list to store trial information
trial_info = []
previous_start_sample = None
processed_starts = set()
start_idx = 0
if subj == 'R2487':
    trail_t = 21
else:
    trail_t = 25
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
            'end_sample': end_sample,
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

epochs_data = []
picks = mne.pick_types(raw.info, meg=True, exclude='bads')

for trial in trial_info:
    end_sample = trial['end_sample']
    start_sample = end_sample - int(sfreq * 10) # 0.2 second before end_sample
    if start_sample < 0:
        start_sample = 0  # Ensure start_sample is not negative

    epoch_data = raw.get_data(start=start_sample, stop=end_sample, picks=picks)
    epochs_data.append(epoch_data)
epochs_array = np.array(epochs_data)
epochs_data_list = []
trial_info_valid = []

for idx, event in enumerate(start_events):
    tmin = -0.2
    tmax = trail_t
    start_sample = event[0]
    event_id_code = event[2]
    total_duration = raw.times[-1]
    print(f"Tmax is {tmax}")
    # Create the epoch
    picks = mne.pick_types(raw.info, meg=True, exclude='bads')
    epochs = mne.Epochs(
        raw, [event], event_id={f'event_{event_id_code}': event_id_code},
        tmin=tmin, tmax=tmax, preload=True,
        reject_by_annotation=False, reject=None, verbose=True, picks=picks
    )
    
    if len(epochs) > 0:
        epochs_data_list.append(epochs.get_data())
        trial_info_valid.append(trial_info[idx])
    else:
        print(f"Epoch {idx} was dropped.")
        print(f"Drop log for Epoch {idx}: {epochs.drop_log}")

def augment_with_sliding_vectors(X, shifts):
    """
    Augments the data by creating shifted versions of the time points.

    Parameters:
    - X: ndarray of shape (n_epochs, n_channels, n_time_points)
    - shifts: list of integers, number of time points to shift (positive or negative).

    Returns:
    - X_augmented: ndarray of shape (n_augmented_epochs, n_channels, n_time_points)
    """
    X_augmented = []
    for shift in shifts:
        if shift > 0:
            # Shift forward
            X_shifted = np.pad(X[:, :, :-shift], ((0, 0), (0, 0), (shift, 0)), mode='constant')
        elif shift < 0:
            # Shift backward
            X_shifted = np.pad(X[:, :, -shift:], ((0, 0), (0, 0), (0, -shift)), mode='constant')
        else:
            # No shift
            X_shifted = X
        X_augmented.append(X_shifted)

    return np.concatenate(X_augmented, axis=0)


def train_time_decoder(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(max_iter=1000))
    time_decoding = SlidingEstimator(clf, n_jobs=5, scoring='accuracy')
    scores = cross_val_multiscore(time_decoding, X, y, cv=cv, n_jobs=5)
    np.save(f'output/{subj}/{subj}_decoding_10ms.npy', scores)
    print(f"Scores shape: {scores.shape}")
    scores_mean = np.mean(scores, axis=0)
    print(f"Scores mean shape: {scores_mean.shape}")
    return scores_mean  


# Convert the filtered epochs data to a numpy array
X = np.array([md.data for md in epochs_array])   # Shape: (n_epochs, n_channels, n_times)
# X = X.squeeze(axis=1)

labels_df = pd.read_csv(f'data_log/{subj}/label.csv')

valid_trial_indices = {info['trial_index'] for info in trial_info_valid}

labels_df_filtered = labels_df[labels_df['trial_index'].isin(valid_trial_indices)]

# Create a mapping from trial_index to label using the filtered labels_df
label_dict = dict(zip(labels_df_filtered['trial_index'], labels_df_filtered['trial.rule']))

y_labels = []
for info in trial_info_valid:
    idx = info['trial_index']
    if idx in label_dict:
        y_labels.append(label_dict[idx])

# Convert labels to integers using label encoder
y_labels = label_encoder.fit_transform(y_labels)

# extractor = EventExtractor(trial_info_valid, raw, label_dict)
# X, y_labels = extractor.extract_events()

shifts = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

# Augment data with surrounding time points
X_augmented = augment_with_sliding_vectors(X, shifts)

y_augmented = np.tile(y_labels, len(shifts))

print(f"Original X shape: {X.shape}, Augmented X shape: {X_augmented.shape}")
print(f"Original y shape: {y_labels.shape}, Augmented y shape: {y_augmented.shape}")

train_time_decoder(X_augmented, y_augmented)