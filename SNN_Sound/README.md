# SNN-Based Blind Source Separation for Speech

## 1. Data Preparation

### Dataset Generation
**Source**:Generated using [LibriMix](https://github.com/JorisCos/LibriMix) toolkit
  - LibriMix is a configurable pipeline for generating speech separation datasets from LibriSpeech corpus
  - Allows customization of mixture parameters (number of speakers, SNR, duration, etc.)

**Specifications**:
  - 15,000 mixed audio samples
  - Each sample contains two speakers talking simultaneously
  - Duration: 3 seconds per audio file
  - Sample rate: 16 kHz
  - Train/Test split: 85% / 15%

### Data Structure
```
LibriMix/output/
├── mix_clean/     # Mixed audio (s1 + s2)
├── s1/            # Source 1 (ground truth)
└── s2/            # Source 2 (ground truth)
```

**Note**: Data generation scripts and complete audio files are not included in this repository due to size constraints.


## 2. Model Architecture

### Signal Processing Pipeline

**Input Processing:**
1. **STFT (Short-Time Fourier Transform)**:
   - Convert time-domain waveform to frequency domain
   - Parameters: `n_fft=512`, `hop_length=160`
   - Output shape: `(batch, 257 freq_bins, time_frames)`

2. **Magnitude Extraction**:
   - Compute magnitude: `|STFT|`
   - Apply log-compression: `log(1 + magnitude)` for stable training

3. **Ground Truth Mask Generation**:
   - Ideal Ratio Mask (IRM): `mask_i = |S_i| / (|S_1| + |S_2| + ε)`
   - Used as training targets

**SNN Architecture:**
```
Input: Log-Magnitude Spectrogram [Batch, 257, Time]
   ↓
Permute → [Batch, Time, 257]
   ↓
Expand to SNN Time Steps T=6 → [T, Batch, Time, 257]
   ↓
Flatten → [T, Batch×Time, 257]
   ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 1: Linear(257 → 512) + BatchNorm + LIF Neuron
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 2: Linear(512 → 512) + BatchNorm + LIF Neuron
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 3: Linear(512 → 514) + LIF Neuron
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ↓
Spike Accumulation over T steps
   ↓
Rate Coding: spike_sum / T → [Batch×Time, 514]
   ↓
Reshape → [Batch, Time, 514]
   ↓
Permute → [Batch, 514, Time]
   ↓
Split → mask1[Batch, 257, Time], mask2[Batch, 257, Time]
   ↓
Output: Two separation masks
```

**Reconstruction:**
1. Apply predicted masks to mixture magnitude: `S_i_mag = |Mix| × mask_i`
2. Combine with mixture phase: `S_i_complex = S_i_mag × exp(j × phase_mix)`
3. Inverse STFT to reconstruct time-domain waveforms

### Loss Function
- **Training Loss**: Permutation Invariant Training (PIT) with MSE on ideal ratio masks
  Handles label ambiguity by choosing the best source permutation
- **Evaluation Metric**: Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) on reconstructed waveforms


### 3. Performance

**Achieved SI-SDR**: **2.23 dB**
**Ideal SI-SDR**: **12.92 dB** (theoretical performance limit)



## 4. Usage
### Installation
```bash
pip install -r requirements.txt
```

### Inference
Separate a mixed audio file into two sources:
```bash
python inference.py --wav  --model  --output 
```

**Example:**
```bash
python inference.py --wav input_mixed_sounds/mix_00000.wav --output output
```

**Arguments:**
- `--wav`: Path to input mixed audio file (required)
- `--model`: Path to trained model checkpoint (default: `./save_models/best_snn.pt`)
- `--output`: Directory to save separated audio files (default: `./output`)

**Output:**
The script generates two separated audio files:
- `<input_name>_s1.wav`: Separated source 1
- `<input_name>_s2.wav`: Separated source 2

---