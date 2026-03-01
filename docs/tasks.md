# Tasks

## Sound Source Separation

- [x] wire output to the speaker in `main.py`
- [x] wire SSN_Sound, and set a minimal apis of arrays to control the gain(since SNN_Sound model would separate the
  audio into multiple pieces, we need to control the gain for each piece), here we hardcode each gain to be 1.
- [x] wire EEG decoding, and replace the gain vector with the EEG decoding output
- [ ] attach sound source separation to real mic
- [ ] attach eeg part to real mic.

## Second Stage: If possible

- [ ] --eeg-real-time cause the overbudget per frame


