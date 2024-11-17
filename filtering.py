import glob
import os
import mne

data_dir = 'data_meg'
save_dir = 'data_meg'
raws = []
subj = "R2487"
exp = 'exp'
dtype = "raw"

def concatenate_raws(subj, dtype):
    reference_dev_head_t = None
    for file in glob.glob(f'{data_dir}/{subj}/{exp}/*{dtype}*.fif'):
        print(file)
        raw = mne.io.read_raw_fif(file, preload=True)
        
        # Set the reference dev_head_t from the first file
        if reference_dev_head_t is None:
            reference_dev_head_t = raw.info['dev_head_t']
        
        # Manually set dev_head_t to the reference
        raw.info['dev_head_t'] = reference_dev_head_t
        raws.append(raw)
    
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
    raw.save(f'{save_dir}/{subj}_{exp}.fif', overwrite=True)
    return raw

def ssp_denoising(subj):
    raw = ica_denoising(subj)
    ssp = mne.preprocessing.SSP(raw)
    ssp.plot_sources(raw)
    input('press enter to continue')
    raw = ssp.apply(raw)
    raw.save(f'{save_dir}/{subj}_{exp}_ssp.fif', overwrite=True)
    return raw

if __name__ == '__main__':
    # concatenate_raws(subj, dtype)
    # bad_channel_interpolation(subj)
    # epoch_data(raw)
    # raw = apply_ica(subj)
    ica_denoising(subj)
    
