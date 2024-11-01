import glob
import os
import mne

data_dir = 'data_meg'
save_dir = 'data_meg'
raws = []
subj = "R2210"
dtype = "raw"

def concatenate_raws(subj, dtype):
    for file in glob.glob(f'{data_dir}/{subj}/*{dtype}*.fif'):
        print(file)
        raws.append(mne.io.read_raw_fif(file, preload=True))
    if raws:  # Check if raws is not empty
        raw = mne.concatenate_raws(raws)
        raw.filter(1, 40, method='iir')
    return raw
    

def preprocess_raws(subj):
    preprocessed_file = f'{data_dir}/{subj}/{subj}_preprocessed.fif'
    if os.path.exists(preprocessed_file):
        print(f"Preprocessed file found: {preprocessed_file}")
        raw = mne.io.read_raw_fif(preprocessed_file, preload=True)
    else:
        print(f"Preprocessed file not found, concatenating raw files.")
        raw = concatenate_raws(subj, dtype)
    return raw

def bad_channel_interpolation(subj):
    # raw = preprocess_raws(subj)
    raw = concatenate_raws(subj, dtype)
    print("Applying highpass and lowpass filter...")
    # Plot and allow marking of bad channels
    raw.plot()
    input('Mark bads, press enter to continue')
    print(f"Bads: {raw.info['bads']}")

    raw.interpolate_bads()
    return raw

def ica_denoising(subj):
    raw = bad_channel_interpolation(subj)
    ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')
    print('fitting ica...')
    ica.fit(raw, reject={'mag': 5e-12})
    ica.plot_sources(raw)
    input('press enter to see topos...')
    ica.plot_components()
    print('excluding:', ica.exclude)
    raw = ica.apply(raw, exclude = ica.exclude)
    raw.save(f'{save_dir}/{subj}_{dtype}_ica.fif', overwrite=True)
    return raw


# def epoch_data(subj,raw):
#     raw = preprocess_raws(subj)
#     events = mne.find_events(raw, stim_channel='STI 014', output='onset', shortest_event=1)
#     event_id = {
#         'start': 1,
#         'move': 2,
#         'reveal_red': 4,
#         'reveal_white': 8,
#         'done': 16,
#     }
#     epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=1.0, baseline=(None, 0), preload=True)
#     epochs.save(f'{save_dir}/{subj}-epo.fif', overwrite=True)
#     return epochs
    
if __name__ == '__main__':
    # concatenate_raws(subj, dtype)
    # bad_channel_interpolation(subj)
    ica_denoising(subj)
    # epoch_data(raw)
    # raw = apply_ica(subj)
    
