import torch
import torch.nn as nn
import torch.nn.functional as F

class MedNeXtBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=5, expansion=4, groups=32):
        super(MedNeXtBlock, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.expansion = expansion

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                        groups=in_channels, padding=kernel_size // 2, bias=False)

        # Group Normalization
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=in_channels)

        # Pointwise convolution (expanding channels by `expansion` factor)
        self.pointwise_conv1 = nn.Conv2d(in_channels, in_channels * expansion, 
                                         kernel_size=1, bias=False)

        # GELU activation
        self.gelu = nn.GELU()

        # Pointwise convolution (reducing channels back to original size)
        self.pointwise_conv2 = nn.Conv2d(in_channels * expansion, in_channels, 
                                         kernel_size=1, bias=False)

    def forward(self, x):
        # Save the input for the residual connection
        residual = x

        # Depthwise convolution
        x = self.depthwise_conv(x)

        # Group Normalization
        x = self.gn(x)

        # First Pointwise convolution (expanding channels)
        x = self.pointwise_conv1(x)

        # GELU activation
        x = self.gelu(x)

        # Second Pointwise convolution (reducing channels)
        x = self.pointwise_conv2(x)

        # Residual connection
        x += residual

        return x

class MedNeXtDown(nn.Module):
    def __init__(self, in_channels, kernel_size=5, expansion=4, groups=32):
        super(MedNeXtDown, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.expansion = expansion

        # Depthwise convolution with stride=2 for downsampling
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                        stride=2, groups=in_channels, padding=kernel_size // 2, 
                                        bias=False)

        # Group Normalization
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=in_channels)

        # Pointwise convolution (expanding channels by `expansion` factor)
        self.pointwise_conv1 = nn.Conv2d(in_channels, in_channels * expansion, 
                                         kernel_size=1, bias=False)

        # GELU activation
        self.gelu = nn.GELU()

        # Pointwise convolution (reducing channels to 2x of the original size)
        self.pointwise_conv2 = nn.Conv2d(in_channels * expansion, in_channels * 2, 
                                         kernel_size=1, bias=False)

        # Residual path: 1x1 convolution with stride=2 to match downsampling
        self.residual_conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, 
                                       stride=2, bias=False)

    def forward(self, x):
        # Save the input for the residual connection
        residual = self.residual_conv(x)

        # Depthwise convolution with stride=2 for downsampling
        x = self.depthwise_conv(x)

        # Group Normalization
        x = self.gn(x)

        # First Pointwise convolution (expanding channels)
        x = self.pointwise_conv1(x)

        # GELU activation
        x = self.gelu(x)

        # Second Pointwise convolution (reducing channels to 2x of the original size)
        x = self.pointwise_conv2(x)

        # Residual connection: add the processed input with the downsampled residual
        x += residual

        return x

class MedNeXtUp(nn.Module):
    def __init__(self, in_channels, kernel_size=5, expansion=4, num_groups=32):
        super(MedNeXtUp, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.expansion = expansion

        # Transpose depthwise convolution with stride=2 for upsampling
        self.trans_depthwise_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, 
                                                       stride=2, groups=in_channels, padding=kernel_size // 2, 
                                                       output_padding=kernel_size % 2, bias=False)

        # Group Normalization
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)

        # Pointwise convolution (expanding channels by `expansion` factor)
        self.pointwise_conv1 = nn.Conv2d(in_channels, in_channels * expansion, 
                                         kernel_size=1, bias=False)

        # GELU activation
        self.gelu = nn.GELU()

        # Pointwise convolution (reducing channels to 1/2 of the original size)
        self.pointwise_conv2 = nn.Conv2d(in_channels * expansion, in_channels // 2, 
                                         kernel_size=1, bias=False)

        # Residual path: Transpose 1x1 convolution with stride=2 to match upsampling
        self.residual_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=1, 
                                                stride=2, output_padding=1, bias=False)

    def forward(self, x):
        # Save the input for the residual connection
        residual = self.residual_conv(x)

        # Transpose depthwise convolution with stride=2 for upsampling
        x = self.trans_depthwise_conv(x)

        # Group Normalization
        x = self.gn(x)

        # First Pointwise convolution (expanding channels)
        x = self.pointwise_conv1(x)

        # GELU activation
        x = self.gelu(x)

        # Second Pointwise convolution (reducing channels to 1/2 of the original size)
        x = self.pointwise_conv2(x)

        # Residual connection: add the processed input with the upsampled residual
        x += residual

        return x

class MedNeXtStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(MedNeXtStem, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class MedNeXtDeepSupervision(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MedNeXtDeepSupervision, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

if __name__ == "__main__":
    # 测试输入: (B, C, H, W) = (1, 64, 224, 224)
    input_tensor = torch.randn(1, 64, 224, 224)

    # 创建MedNeXtDown实例，设定扩展倍数为4
    down_block = MedNeXtDown(in_channels=64, expansion=4)

    # 通过MedNeXtDown模块
    output = down_block(input_tensor)

    # 打印输出维度
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")


    # 测试输入: (B, C, H, W) = (1, 64, 224, 224)
    input_tensor = torch.randn(1, 64, 224, 224)

    # 创建MedNeXtBlock实例，设定扩展倍数为4
    block = MedNeXtBlock(in_channels=64, expansion=4)

    # 通过MedNeXtBlock模块
    output = block(input_tensor)

    # 打印输入和输出维度
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # 测试输入: (B, C, H, W) = (1, 64, 112, 112)
    input_tensor = torch.randn(1, 64, 112, 112)

    # 创建MedNeXtUp实例，设定扩展倍数为4
    up_block = MedNeXtUp(in_channels=64, expansion=4)

    # 通过MedNeXtUp模块
    output = up_block(input_tensor)

    # 打印输入和输出维度
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # 假设输入: (B, C, H, W) = (1, 256, 28, 28)
    input_tensor = torch.randn(1, 256, 28, 28)

    # 创建MedNeXtDeepSupervision实例，设定num_classes为10
    deep_supervision = MedNeXtDeepSupervision(in_channels=256, num_classes=10)

    # 通过MedNeXtDeepSupervision模块
    output = deep_supervision(input_tensor)

    # 打印输入和输出维度
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")