# Current Problems and Adapter Scope

## Context
- `earcode` model expects EEG input shaped as `128 x 20` per decision window.
- Real device stream from `cyton_usb_reader.py` in 16-channel mode provides `125 Hz, 16 channels`.

## Problems Found
1. Sampling-rate mismatch
- Training/inference config in `earcode` uses 128 Hz windows.
- Cyton + Daisy real-time stream is 125 Hz for combined 16-channel samples.

2. Channel-count mismatch
- `earcode` pipeline is built for 20 channels.
- Cyton real-time data provides 16 EEG channels.

3. Channel-definition mismatch (legacy dataset vs real device)
- Existing `EAR_4_direction_1D.mat` only contains `EEG` and `ENV`.
- There is no channel-name/electrode mapping metadata in that file to prove semantic alignment with Cyton channels.

4. Real-time integration gap
- Original workflow was offline-focused.
- No direct real-time bridge from Cyton stream to `earcode` model input.

## What the Adapter Solves (in `eeg_main.py`)
1. Real-time ingestion
- `--real-time` mode now reads live data via `cyton_usb_reader.py` logic.

2. Rate adaptation
- Converts one 1-second block from `125 x 16` to `128 x 16` using linear interpolation.

3. Channel adaptation
- Expands `128 x 16` to `128 x 20` by zero-padding channels 17-20.

4. Immediate model compatibility
- Adapted window is fed to the existing `earcode` checkpoint without changing `main.py`.
- Produces real-time 4-class output vectors through `eeg_get_signal()`.

## Important Limitations (Not Solved by Adapter)
1. Not a calibrated channel mapping
- Zero-padding does not reconstruct missing sensor information.
- This is a compatibility bridge, not a physiologically valid channel remap.

2. Accuracy risk
- Current checkpoint was trained on a different data distribution and channel setup.
- Real-time predictions can be biased or unstable.

3. Production readiness
- For reliable performance, retraining is still required with data that matches deployment conditions
  (same channel layout, same sampling strategy, same preprocessing).

## Practical Status
- Adapter enables a working end-to-end demo under time constraints.
- It resolves interface mismatch issues (shape/rate/runtime path) so the full pipeline can run now.
