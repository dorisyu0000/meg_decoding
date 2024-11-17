# meg_decoding

## Pre-processing
The NYU/KIT MEG machine outputs data in .sqd format, but .fif is most standard. The very first thing to do is to download a special GUI for this: [mne-kit-gui](https://github.com/mne-tools/mne-kit-gui).  

```bash
$ conda activate meg
$ mne kit2fiff
```
1. After this, the gui should appear. You will then need to upload the following:
    - Source Markers 1 ( *.mrk), collected at the beginning of the scan
    - Source Markers 2 (*.mrk), collected at the end of the scan
    - The MEG data (*.sqd)
    - The digitizer head shape (*.txt). Make sure this was down-sampled on the computer in the MEG room
    - The digitizer fiducials (*.txt)

2. You will then need to make sure that the markers and fiducial points are lined up. The easiest way to do this is to include only points that line up with the light blue marker (which is a transformation between the two source markers). At any rate, this is most important for source localization, so if you don’t care about that it doesn’t really matter.

3. Once you are finished, you need to make sure that the Events (the events of interest from your task as indicated by the triggers), are being included properly. If you used Psychopy, this is typically “Peak”. To check, click “Find Events”, and make sure the correct number of events show up. You can choose whichever value coding you like; this just changes how each trigger is labeled in the data.

4. Repeat this for all of your .sqd files to create a .fif for each.

## Denoising 
This includes band-pass filtering, interpolating bad channels, ICA, and SSP.
```bash
$ conda activate meg
$ python filtering.py
```

## Decoding

In the decoding.ipynb.