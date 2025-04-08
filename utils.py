import numpy as np
import mne

class EventExtractor:
    def __init__(self, trial_info_valid, raw, label_dict):
        self.trial_info_valid = trial_info_valid
        self.raw = raw
        self.label_dict = label_dict
        self.sfreq = raw.info['sfreq']  # Sampling frequency
        self.picks = mne.pick_types(raw.info, meg=True, exclude='bads')

    def extract_reveal(self, event_name='reveal_red', n_points_before=50, n_points_after=50, num_events=3):
        y_labels = []
        X_reveal = []
        trial_indices = []
        sfreq = self.raw.info['sfreq']  # Sampling frequency

        for info in self.trial_info_valid:
            reveal_times = info[f'{event_name}_times']
            if len(reveal_times) >= num_events:
                trial_data = []

                for event_time in reveal_times[-num_events:]:
                    event_sample = int(event_time * sfreq)
                    start_sample_before = event_sample - n_points_before
                    end_sample_after = event_sample + n_points_after

                    if start_sample_before >= 0 and end_sample_after <= self.raw.n_times:
                        picks = mne.pick_types(self.raw.info, meg=True, exclude='bads')
                        epoch_data = self.raw.get_data(start=start_sample_before, stop=end_sample_after,picks=picks)
                    
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
                        if trial_index in self.label_dict:
                            y_labels.append(self.label_dict[trial_index])
                            trial_indices.append(trial_index)
                    else:
                        print(f"Inconsistent shapes in trial data for trial index {info['trial_index']}: {trial_data_shapes}")

        return np.array(X_reveal), np.array(y_labels), trial_indices

    def extract_start(self, n_points_before=50, n_points_after=50):
        X_start = []
        y_labels = []
        trial_indices = []
        for info in self.trial_info_valid:
            start_sample = info['event_sample']
            start_sample_before = start_sample - n_points_before
            end_sample_after = start_sample + n_points_after

            # Ensure the samples are within bounds
            if start_sample_before >= 0 and end_sample_after <= self.raw.n_times:
                # Extract data for this trial
                picks = mne.pick_types(self.raw.info, meg=True, exclude='bads')
                epoch_data = self.raw.get_data(start=start_sample_before, stop=end_sample_after,picks=picks)
                X_start.append(epoch_data)
                trial_index = info['trial_index']
                if trial_index in self.label_dict:
                    y_labels.append(self.label_dict[trial_index])
                    trial_indices.append(trial_index)

        return np.array(X_start), np.array(y_labels), trial_indices


    def extract_done(self, n_points_before=50, n_points_after=50):
        X_done = []
        y_labels = []
        trial_indices = []
        sfreq = self.raw.info['sfreq']  # Sampling frequency
        done_times = [info['done_times'] for info in self.trial_info_valid]
        start_times = [info['start_times'] for info in self.trial_info_valid]
        
        for info, done_time, start_time in zip(self.trial_info_valid, done_times, start_times):
            # Check if the "done" event exists
            done_time = info['done_times'] 
            start_time = info['start_times']
            
            if int(done_time) - int(start_time) > 25:  # Assuming these indicate the presence of a "done" event
                done_sample = info['event_sample'] + int((info['tmax'] - 1.0) * sfreq)
                start_sample_before = done_sample - n_points_before
                end_sample_after = done_sample + n_points_after
            else:
                start_sample_before = int(done_time * sfreq) - n_points_before
                end_sample_after = int(done_time * sfreq) + n_points_after

            # Ensure the samples are within bounds
            if start_sample_before >= 0 and end_sample_after <= self.raw.n_times:
                picks = mne.pick_types(self.raw.info, meg=True, exclude='bads')
                epoch_data = self.raw.get_data(start=start_sample_before, stop=end_sample_after,picks=picks)
                    
                X_done.append(epoch_data)
                trial_index = info['trial_index']
                if trial_index in self.label_dict:
                    y_labels.append(self.label_dict[trial_index])
                    trial_indices.append(trial_index)

        return np.array(X_done), np.array(y_labels), trial_indices

    def filter_by_trial_indices(self, epochs, labels, trial_indices, common_indices):
        mask = [i for i, idx in enumerate(trial_indices) if idx in common_indices]
        return epochs[mask], labels[mask]

    def extract_events(self):
        epoch_reveal, y_labels_reveal, trial_indices_reveal = self.extract_reveal()
        epoch_start, y_labels_start, trial_indices_start = self.extract_start()
        epoch_done, y_labels_done, trial_indices_done = self.extract_done()
        print(len(trial_indices_reveal), len(trial_indices_start), len(trial_indices_done))
        print(epoch_reveal.shape, epoch_start.shape, epoch_done.shape)
        common_trial_indices = set(trial_indices_reveal) & set(trial_indices_start) & set(trial_indices_done)
        common_trial_indices = sorted(common_trial_indices)

        epoch_start_filtered, y_labels_start_filtered = self.filter_by_trial_indices(epoch_start, y_labels_start, trial_indices_start, common_trial_indices)
        epoch_reveal_filtered, y_labels_reveal_filtered = self.filter_by_trial_indices(epoch_reveal, y_labels_reveal, trial_indices_reveal, common_trial_indices)
        epoch_done_filtered, y_labels_done_filtered = self.filter_by_trial_indices(epoch_done, y_labels_done, trial_indices_done, common_trial_indices)

        n_trials, n_channels, n_timepoints_per_event, n_events = epoch_reveal_filtered.shape
        epoch_reveal_flattened = epoch_reveal_filtered.reshape(n_trials, n_channels, n_timepoints_per_event * n_events)

        X_combined = np.concatenate((epoch_start_filtered, epoch_reveal_flattened, epoch_done_filtered), axis=2)
        y_combined = y_labels_done_filtered

        return X_combined, y_combined

# Example usage:
# extractor = EventExtractor(trial_info_valid, raw, label_dict)
# X_combined, y_combined = extractor.extract_events()
