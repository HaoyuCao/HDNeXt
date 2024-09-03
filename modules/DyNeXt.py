import torch
import torch.nn as nn
import torch.nn.functional as F

class MedNeXt2D(nn.Module):
    def __init__(self, in_channels):
        super(MedNeXt2D, self).__init__()
        
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            groups=in_channels,  # Depth-wise convolution
            bias=False
        )
        
        self.pointwise_conv_expand = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=4 * in_channels,  # Expand channels to 4x
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        
        self.gelu = nn.GELU()
        
        self.pointwise_conv_shrink = nn.Conv2d(
            in_channels=4 * in_channels, 
            out_channels=in_channels,  # Shrink channels back to original
            kernel_size=1, 
            stride=1, 
            bias=False
        )
        
        self.residual_connect = nn.Identity()
        
    def forward(self, x):
        # Input
        identity = self.residual_connect(x)
        
        # 5x5 Depth-wise Convolution
        out = self.depthwise_conv(x)
        
        # 1x1 Point-wise Convolution (Expand Channels)
        out = self.pointwise_conv_expand(out)
        
        # GELU Activation
        out = self.gelu(out)
        
        # 1x1 Point-wise Convolution (Shrink Channels)
        out = self.pointwise_conv_shrink(out)
        
        # Add residual (skip connection)
        out += identity
        
        return out
    
class GRN(nn.Module):
    def __init__(self, num_channels):
        super(GRN, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, X):
        gx = torch.norm(X, p=2, dim=(2, 3), keepdim=True)  # L2 norm along spatial dimensions (H, W)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)    # Normalize across the channel dimension (C)
        return self.gamma * (X * nx) + self.beta + X       # Affine transform and residual connection

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DyNeXt(nn.Module):
    def __init__(self, in_channels, K=5):
        super(DyNeXt, self).__init__()
        self.K = K
        self.in_channels = in_channels
        
        # Adaptive Global Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected Layers in Branch 1
        self.fc1 = nn.Linear(in_channels, in_channels // 4, bias=False)
        self.ln1 = LayerNorm(in_channels // 4, data_format="channels_first")
        self.fc2 = nn.Linear(in_channels // 4, in_channels * K * K, bias=False)
        
        # LayerNorm (only applied on channel dimension)
        self.ln2 = LayerNorm(in_channels, data_format="channels_first")
        
        # Point-wise Convolutions
        self.expand_conv = nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1, bias=False)
        self.shrink_conv = nn.Conv2d(4 * in_channels, in_channels, kernel_size=1, bias=False)
        
        # Activation
        self.gelu = nn.GELU()
        
        # GRN Module
        self.grn = GRN(4 * in_channels)
        
        # Residual Connection
        self.residual_connect = nn.Identity()
        
    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Branch 1: Adaptive Global Pooling -> Fully Connected -> Reshape to Depth-wise Kernel
        out = self.pool(x)
        out = out.view(out.size(0), -1)  # Flatten for fully connected layer
        out = self.fc1(out)
        out = self.ln1(out.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # Apply LayerNorm
        out = self.fc2(out)
        dynamic_weights = out.view(batch_size * self.in_channels, 1, self.K, self.K)
        
        # Branch 2: Depth-wise Convolution using dynamic weights
        x_reshaped = x.view(1, batch_size * self.in_channels, height, width)
        out = torch.nn.functional.conv2d(x_reshaped, weight=dynamic_weights, groups=batch_size * self.in_channels, padding=self.K//2)
        out = out.view(batch_size, self.in_channels, height, width)
        
        # Branch 3: LayerNorm -> Point-wise Conv (Expand) -> GELU -> GRN -> Point-wise Conv (Shrink)
        out = self.ln2(out)
        out = self.expand_conv(out)
        out = self.gelu(out)
        out = self.grn(out)
        out = self.shrink_conv(out)
        
        # Residual Connection: Add input to the output
        out += self.residual_connect(x)
        
        return out

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# 假设 GRN 模块也已经定义
class GRN(nn.Module):
    def __init__(self, num_channels):
        super(GRN, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x