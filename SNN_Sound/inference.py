import os
import argparse
import torch
import numpy as np
from scipy.io import wavfile
from snn_model import SimpleSNN
from spikingjelly.clock_driven import functional

def separate_audio(wav_path, model_path, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(output_dir, exist_ok=True)
    
    model = SimpleSNN(input_size=257, hidden_size=512, T=6).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    sr, audio = wavfile.read(wav_path)
    
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / 32768.0

    mix_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    
    N_FFT = 512
    HOP_LENGTH = 160
    win = torch.hann_window(N_FFT).to(device)

    with torch.no_grad():
        mix_stft = torch.stft(mix_tensor, N_FFT, HOP_LENGTH, return_complex=True, window=win)
        mix_mag = torch.abs(mix_stft)
        mix_phase = torch.angle(mix_stft)
        mix_input = torch.log1p(mix_mag)

        functional.reset_net(model)
        m1_pred, m2_pred = model(mix_input)

        est_mag1 = mix_mag * m1_pred
        est_mag2 = mix_mag * m2_pred

        est_s1 = torch.istft(est_mag1 * torch.exp(1j * mix_phase), N_FFT, HOP_LENGTH, window=win)
        est_s2 = torch.istft(est_mag2 * torch.exp(1j * mix_phase), N_FFT, HOP_LENGTH, window=win)

    s1_np = est_s1.squeeze().cpu().numpy()
    s2_np = est_s2.squeeze().cpu().numpy()

    orig_len = len(audio)
    s1_np = s1_np[:orig_len]
    s2_np = s2_np[:orig_len]

    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    
    out_path1 = os.path.join(output_dir, f'{base_name}_s1.wav')
    out_path2 = os.path.join(output_dir, f'{base_name}_s2.wav')

    wavfile.write(out_path1, sr, s1_np)
    wavfile.write(out_path2, sr, s2_np)

    print(f"Processed: {wav_path}")
    print(f"Model used: {model_path}")
    print(f"Saved s1: {out_path1}")
    print(f"Saved s2: {out_path2}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Required argument
    parser.add_argument('--wav', type=str, required=True, help='Path to input wav file')
    
    # Optional arguments with defaults
    parser.add_argument('--model', type=str, default='./save_models/best_snn.pt', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./output', help='Directory to save results')
    
    args = parser.parse_args()
    
    separate_audio(args.wav, args.model, args.output)