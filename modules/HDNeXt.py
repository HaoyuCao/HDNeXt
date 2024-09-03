import torch
import torch.nn as nn
from DyNeXt import DyNeXt, MedNeXt2D

class HDNeXt(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(HDNeXt, self).__init__()
        
        # Initial convolution layer to convert input channels to 64
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # DyNeXt Block 1 & 2 (通道数保持为64)
        self.dynext_block1 = DyNeXt(in_channels=base_channels, K=5, shrink_mode='full')
        self.dynext_block2 = DyNeXt(in_channels=base_channels, K=5, shrink_mode='full')
        
        # Downsample 1 + DyNeXt Block 3 (通道扩展到128)
        self.downsample1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.dynext_block3 = DyNeXt(in_channels=base_channels, K=5, shrink_mode='half')
        
        # DyNeXt Block 4 (通道数保持为128)
        self.dynext_block4 = DyNeXt(in_channels=base_channels * 2, K=5, shrink_mode='full')
        
        # Downsample 2 + DyNeXt Block 5 (通道扩展到256)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.dynext_block5 = DyNeXt(in_channels=base_channels * 2, K=5, shrink_mode='half')
        
        # DyNeXt Block 6 (通道数保持为256)
        self.dynext_block6 = DyNeXt(in_channels=base_channels * 4, K=5, shrink_mode='full')

    def forward(self, x):
        # Initial Convolution + ReLU
        x = self.initial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # DyNeXt Block 1 & 2
        x = self.dynext_block1(x)
        print(f"Stage 1 输出形状: {x.shape}")
        x = self.dynext_block2(x)
        print(f"Stage 2 输出形状: {x.shape}")
        
        # Downsample 1 + DyNeXt Block 3 (通道扩展)
        x = self.downsample1(x)
        x = self.dynext_block3(x)
        print(f"Stage 3 输出形状: {x.shape}")
        
        # DyNeXt Block 4
        x = self.dynext_block4(x)
        print(f"Stage 4 输出形状: {x.shape}")
        
        # Downsample 2 + DyNeXt Block 5 (通道扩展)
        x = self.downsample2(x)
        x = self.dynext_block5(x)
        print(f"Stage 5 输出形状: {x.shape}")
        
        # DyNeXt Block 6
        x = self.dynext_block6(x)
        print(f"Stage 6 输出形状: {x.shape}")
        
        return x