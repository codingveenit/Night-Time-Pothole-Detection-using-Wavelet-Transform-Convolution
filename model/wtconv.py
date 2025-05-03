import torch
import torch.nn as nn
import pywt
import numpy as np

class WTConv(nn.Module):
    def __init__(self, in_channels, out_channels, wavelet='db1'):
        super(WTConv, self).__init__()
        self.wavelet = wavelet
        self.conv = nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        coeffs_list = []
        
        for i in range(batch_size):
            batch_coeffs = []
            for c in range(channels):
                # Get numpy array for wavelet transform
                img = x[i, c].cpu().detach().numpy()
                
                # Apply wavelet transform
                coeffs = pywt.dwt2(img, self.wavelet)
                LL, (LH, HL, HH) = coeffs
                
                # Stack the coefficients as channels
                batch_coeffs.append(torch.from_numpy(np.stack([LL, LH, HL, HH])))
            
            # Concatenate all channel coefficients
            coeffs_tensor = torch.cat(batch_coeffs, dim=0)
            coeffs_list.append(coeffs_tensor)
        
        # Stack all batches
        x_transformed = torch.stack(coeffs_list).to(x.device).float()
        output = self.conv(x_transformed)
        output = self.bn(output)
        output = self.act(output)
        
        return output
