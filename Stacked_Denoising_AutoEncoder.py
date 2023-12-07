import torch
import torch.nn as nn

# Stacked Denoising AE
class StackedDenoisingAE(nn.Module):
    def __init__(self, input_size, output_size):
        super(StackedDenoisingAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_size, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, output_size, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(output_size, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.Sigmoid() # 각 픽셀을 0-1 값으로 변환해야함 
        )
        
    def forward(self, x):
        en_out_x = self.encoder(x)
        de_out_x = self.decoder(en_out_x) 
        return de_out_x