import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
from sklearn.model_selection import train_test_split


class SpeechSeparationDataset(Dataset):
    """Dataset for speech separation with mixture and two sources"""
    def __init__(self, data_dir, indices, duration=3.0, sample_rate=16000):
        """
        Args:
            data_dir: root directory containing mix_clean, s1, s2 folders
            indices: list of file indices to load (e.g., [0, 1, 2, ...])
            duration: audio duration in seconds
            sample_rate: sampling rate
        """
        self.data_dir = data_dir
        self.indices = indices
        self.duration = duration
        self.sample_rate = sample_rate
        self.target_length = int(duration * sample_rate)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Try to find a valid file, maximum 10 different files
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                file_idx = self.indices[(idx + attempt) % len(self)]
                
                # Construct file paths
                mix_path = os.path.join(self.data_dir, 'mix_clean', f'mix_{file_idx:05d}.wav')
                s1_path = os.path.join(self.data_dir, 's1', f'mix_{file_idx:05d}.wav')
                s2_path = os.path.join(self.data_dir, 's2', f'mix_{file_idx:05d}.wav')
                
                # Load audio files using scipy (more stable than torchaudio)
                sr_mix, mixture = wavfile.read(mix_path)
                sr_s1, source1 = wavfile.read(s1_path)
                sr_s2, source2 = wavfile.read(s2_path)
                
                # Convert to float32 and normalize
                mixture = mixture.astype(np.float32) / 32768.0
                source1 = source1.astype(np.float32) / 32768.0
                source2 = source2.astype(np.float32) / 32768.0
                
                # Convert to torch tensors
                mixture = torch.from_numpy(mixture)
                source1 = torch.from_numpy(source1)
                source2 = torch.from_numpy(source2)
                
                # Convert to mono if stereo
                if mixture.dim() > 1:
                    mixture = mixture.mean(dim=-1)
                if source1.dim() > 1:
                    source1 = source1.mean(dim=-1)
                if source2.dim() > 1:
                    source2 = source2.mean(dim=-1)
                
                # Fix length (pad or trim)
                mixture = self._fix_length(mixture.unsqueeze(0)).squeeze(0)
                source1 = self._fix_length(source1.unsqueeze(0)).squeeze(0)
                source2 = self._fix_length(source2.unsqueeze(0)).squeeze(0)
                
                return mixture, source1, source2
                
            except Exception as e:
                if attempt == 0:
                    print(f"\nSkipping file {file_idx:05d}: {e}")
                continue
        
        # If all attempts failed, raise error
        raise RuntimeError(f"Failed to load any valid file after {max_attempts} attempts")
    
    def _fix_length(self, audio):
        """Pad or trim audio to target length"""
        # audio shape: (1, time)
        if audio.shape[1] < self.target_length:
            # Pad
            pad = self.target_length - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad))
        elif audio.shape[1] > self.target_length:
            # Trim
            audio = audio[:, :self.target_length]
        return audio


def create_dataloaders(data_dir, batch_size=8, test_split=0.15, num_workers=0, seed=42):
    """
    Create train and test dataloaders
    
    Args:
        data_dir: root directory (e.g., './output')
        batch_size: batch size for training
        test_split: fraction of data for testing (0.15 = 15%)
        num_workers: number of workers for data loading
        seed: random seed for reproducibility
        
    Returns:
        train_loader, test_loader, train_size, test_size
    """
    # Get all files by listing directory
    mix_dir = os.path.join(data_dir, 'mix_clean')
    files = os.listdir(mix_dir)
    
    # Extract indices from filenames
    all_indices = []
    for f in files:
        if f.startswith('mix_') and f.endswith('.wav'):
            idx = int(f[4:9])  # extract 00000 from mix_00000.wav
            all_indices.append(idx)
    
    all_indices.sort()
    print(f"Found {len(all_indices)} valid files")
    
    if len(all_indices) == 0:
        raise RuntimeError("No valid wav files found!")
    
    # Train/test split
    train_indices, test_indices = train_test_split(
        all_indices, 
        test_size=test_split, 
        random_state=seed
    )
    
    # Create datasets
    train_dataset = SpeechSeparationDataset(data_dir, train_indices)
    test_dataset = SpeechSeparationDataset(data_dir, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader, len(train_indices), len(test_indices)


def compute_metrics(s1_pred, s2_pred, s1_true, s2_true, mel_transform):
    """
    Compute separation metrics
    
    Args:
        s1_pred, s2_pred: predicted mel spectrograms (batch, n_mels, time)
        s1_true, s2_true: target waveforms (batch, time)
        mel_transform: mel spectrogram transform
        
    Returns:
        dict with metrics
    """
    from snn_model import compute_si_sdr
    
    # Convert true sources to mel spectrograms
    with torch.no_grad():
        s1_mel_true = mel_transform(s1_true)
        s1_mel_true = torch.log(s1_mel_true + 1e-8)
        s1_mel_true = (s1_mel_true - s1_mel_true.mean(dim=(1,2), keepdim=True)) / (s1_mel_true.std(dim=(1,2), keepdim=True) + 1e-8)
        
        s2_mel_true = mel_transform(s2_true)
        s2_mel_true = torch.log(s2_mel_true + 1e-8)
        s2_mel_true = (s2_mel_true - s2_mel_true.mean(dim=(1,2), keepdim=True)) / (s2_mel_true.std(dim=(1,2), keepdim=True) + 1e-8)
    
    # Compute SI-SDR for both sources
    si_sdr1 = compute_si_sdr(s1_pred, s1_mel_true)
    si_sdr2 = compute_si_sdr(s2_pred, s2_mel_true)
    avg_si_sdr = (si_sdr1 + si_sdr2) / 2
    
    # Compute MSE
    mse1 = torch.nn.functional.mse_loss(s1_pred, s1_mel_true)
    mse2 = torch.nn.functional.mse_loss(s2_pred, s2_mel_true)
    total_mse = mse1 + mse2
    
    return {
        'si_sdr1': si_sdr1.item(),
        'si_sdr2': si_sdr2.item(),
        'avg_si_sdr': avg_si_sdr.item(),
        'mse_loss': total_mse.item()
    }


if __name__ == '__main__':
    # Test data loading
    data_dir = r'./LibriMix/output'
    
    train_loader, test_loader, train_size, test_size = create_dataloaders(
        data_dir, batch_size=4
    )
    
    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    mixture, source1, source2 = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Mixture: {mixture.shape}")
    print(f"Source 1: {source1.shape}")
    print(f"Source 2: {source2.shape}")