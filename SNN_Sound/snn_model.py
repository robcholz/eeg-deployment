import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, surrogate

class SimpleSNN(nn.Module):
    def __init__(self, input_size=257, hidden_size=512, T=6): 
        super().__init__()
        self.T = T

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size), 
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size), 
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, input_size * 2),
            
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        )

    def forward(self, x):
        
        x = x.permute(0, 2, 1) 
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1) 
        spike_sum = 0
        B, Time, F = x.shape

        x_flat = x_seq.view(self.T, B * Time, F)
        
        for t in range(self.T):
            out1 = self.layer1(x_flat[t])
            out2 = self.layer2(out1)
            out3 = self.layer3(out2) 
            
            spike_sum += out3 

        mask_out = spike_sum / self.T

        mask_out = mask_out.view(B, Time, -1)

        mask_out = mask_out.permute(0, 2, 1)
        
        freq_dim = mask_out.shape[1] // 2
        mask1 = mask_out[:, :freq_dim, :]
        mask2 = mask_out[:, freq_dim:, :]
        
        return mask1, mask2