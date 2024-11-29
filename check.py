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
from util import EventExtractor
from sklearn.decomposition import PCA
# # Parse command-line arguments
# parser = argparse.ArgumentParser(description='Process MEG data for a specific subject.')
# parser.add_argument('--subject', type=str, required=True, help='Subject identifier')
# args = parser.parse_args()

# # Preprocess data
# subj = args.subject
subj = 'R2490'
data_dir = 'data_meg'
dataqual = 'prepro' #or loc/exp
exp = 'exp' #or exp
dtype = "raw"
label_dir = 'data_log'
save_dir = 'data_meg'
raw = mne.io.read_raw_fif(f'{data_dir}/{subj}/{dataqual}/{subj}_{exp}.fif', preload=True)
# raw = mne.io.read_raw_fif('data_meg/R2210/prepro/R2210_raw_ica.fif',preload=True)
bad_channels_dict = {
    "R2210": ['MEG 079', 'MEG 076', 'MEG 068', 'MEG 015','MEG 014','MEG 147'],
    "R2490": ['MEG 014', 'MEG 004', 'MEG 079', 'MEG 072', 'MEG 070', 'MEG 080', 'MEG 074', 'MEG 067', 'MEG 082', 'MEG 105', 'MEG 115', 'MEG 141', 'MEG 153'],
    "R2488": ['MEG 015', 'MEG 014', 'MEG 068', 'MEG 079', 'MEG 146', 'MEG 147', 'MEG 007', 'MEG 141'],
    "R2487": ['MEG 015', 'MEG 014', 'MEG 068', 'MEG 079', 'MEG 147', 'MEG 146', 'MEG 004'],
    "R2280": ['MEG 024', 'MEG 039', 'MEG 079', 'MEG 077', 'MEG 141', 'MEG 073', 'MEG 075', 'MEG 076', 'MEG 064', 'MEG 063', 'MEG 060', 'MEG 059', 'MEG 058']
}
bad_channels = bad_channels_dict.get(subj, [])
raw.info['bads'].extend(bad_channels)
raw.drop_channels(bad_channels)
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


# Initialize lists to store epochs and labels
data_list = []
labels = []

# Iterate through each start event to create trial information
for idx, start_event in enumerate(start_events):
    start_sample = start_event[0] - 50
    
    # Define tmin and tmax for 1 second before and after the start event
    tmin_before = -0.5 # 1 second before start
    tmax_before = 0.0  # up to the start
    tmin_after = 0.0    # from the start
    tmax_after = 0.5    # 1 second after start
    picks = mne.pick_types(raw.info, meg=True, exclude='bads')
    # Extract epochs for 1 second before the start event
    epochs = mne.Epochs(
        raw, [start_event], tmin=tmin_before, tmax=tmax_before, preload=True,
        picks=picks, reject_by_annotation=False, reject=None, verbose=False, baseline=(0, 0)
    )
    data_list.append(epochs.get_data())
    

X = np.array([md.data for md in data_list])  # Ensure this is 3D
X = X.squeeze(axis=1)
X_reshaped = X.transpose(2, 0, 1).reshape(391, -1) 
y = np.array([0 if t < 0 else 1 for t in epochs.times])

pca = PCA(n_components=15)  # Reduce to 10 components
X_pca = pca.fit_transform(X_reshaped)

clf = make_pipeline(
    PCA(n_components=10),  # Dimensionality reduction
    LogisticRegression(solver="liblinear")  # Binary classification
)

# Use Leave-One-Out cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
accuracy = cross_val_score(clf, X, y, cv=cv).mean()

print(f"Decoding accuracy for {subj}: {accuracy:.2f}")