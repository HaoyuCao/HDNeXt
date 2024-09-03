from MedNeXt_Block import MedNeXtStem, MedNeXtDown, MedNeXtBlock, MedNeXtDeepSupervision, MedNeXtUp
import torch
import torch.nn as nn
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count_table


class MedNeXtEncoder(nn.Module):
    def __init__(self, in_channels=3, C=64, encoder_blocks=[1,1,1,1], encoder_expansion=[4,4,4,4]):
        super(MedNeXtEncoder, self).__init__()
        self.C = C
        assert len(encoder_blocks) == 4, "encoder_blocks should have 4 elements."
        assert len(encoder_expansion) == 4, "encoder_expansion should have 4 elements."
        
        # Stem
        self.stem = MedNeXtStem(in_channels=in_channels, out_channels=C)
        
        # Encoder layers
        self.encoder1 = self._make_layer(MedNeXtBlock, C, encoder_blocks[0], encoder_expansion[0])
        self.encoder2 = nn.Sequential(
            MedNeXtDown(in_channels=C, expansion=encoder_expansion[0]),
            self._make_layer(MedNeXtBlock, 2*C, encoder_blocks[1], encoder_expansion[1])
        )
        self.encoder3 = nn.Sequential(
            MedNeXtDown(in_channels=2*C, expansion=encoder_expansion[1]),
            self._make_layer(MedNeXtBlock, 4*C, encoder_blocks[2], encoder_expansion[2])
        )
        self.encoder4 = nn.Sequential(
            MedNeXtDown(in_channels=4*C, expansion=encoder_expansion[2]),
            self._make_layer(MedNeXtBlock, 8*C, encoder_blocks[3], encoder_expansion[3])
        )

    def _make_layer(self, block, out_channels, blocks, expansion):
        layers = []
        for _ in range(blocks):
            layers.append(block(out_channels, expansion=expansion))
        return nn.Sequential(*layers)

    def forward(self, x):
        skip_connections = []

        # Stem
        x = self.stem(x)  # (B, C, H, W)
        
        # Encoder layers
        x = self.encoder1(x)  # (B, C, H, W)
        skip_connections.append(x)
        
        x = self.encoder2(x)  # (B, 2C, H/2, W/2)
        skip_connections.append(x)
        
        x = self.encoder3(x)  # (B, 4C, H/4, W/4)
        skip_connections.append(x)
        
        x = self.encoder4(x)  # (B, 8C, H/8, W/8)
        skip_connections.append(x)

        return x, skip_connections[::-1]  # Reverse the skip connections for the decoder


class MedNeXtDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, decoder_blocks, decoder_expansion, num_classes):
        super(MedNeXtDecoder, self).__init__()

        # Decoder stages (including the final segmentation head stage)
        self.decoder_stages = nn.ModuleList([
            nn.Sequential(
                self._make_layer(MedNeXtBlock, in_channels, decoder_blocks[0], decoder_expansion[0]),
                MedNeXtUp(in_channels=in_channels, expansion=decoder_expansion[0])
            ),
            nn.Sequential(
                self._make_layer(MedNeXtBlock, skip_channels[0], decoder_blocks[1], decoder_expansion[1]),
                MedNeXtUp(in_channels=skip_channels[0], expansion=decoder_expansion[1])
            ),
            nn.Sequential(
                self._make_layer(MedNeXtBlock, skip_channels[1], decoder_blocks[2], decoder_expansion[2]),
                MedNeXtUp(in_channels=skip_channels[1], expansion=decoder_expansion[2])
            ),
            nn.Sequential(
                self._make_layer(MedNeXtBlock, skip_channels[2], decoder_blocks[3], decoder_expansion[3]),
                MedNeXtUp(in_channels=skip_channels[2], expansion=decoder_expansion[3])
            ),
            # Final segmentation head stage
            nn.Sequential(
                self._make_layer(MedNeXtBlock, skip_channels[3], decoder_blocks[4], decoder_expansion[4]),
                MedNeXtDeepSupervision(in_channels=skip_channels[3], num_classes=num_classes)
            )
        ])
        
        # 1x1 convolution to reduce channels after concatenation
        self.conv1x1_layers = nn.ModuleList([
            nn.Conv2d(skip_channels[0] * 2, skip_channels[0], kernel_size=1, bias=False),
            nn.Conv2d(skip_channels[1] * 2, skip_channels[1], kernel_size=1, bias=False),
            nn.Conv2d(skip_channels[2] * 2, skip_channels[2], kernel_size=1, bias=False),
            nn.Conv2d(skip_channels[3] * 2, skip_channels[3], kernel_size=1, bias=False)
        ])
        
        # Deep Supervision layers for intermediate outputs
        self.deep_supervision_layers = nn.ModuleList([
            MedNeXtDeepSupervision(in_channels=skip_channels[0], num_classes=num_classes),
            MedNeXtDeepSupervision(in_channels=skip_channels[1], num_classes=num_classes),
            MedNeXtDeepSupervision(in_channels=skip_channels[2], num_classes=num_classes),
            MedNeXtDeepSupervision(in_channels=skip_channels[3], num_classes=num_classes)
        ])

    def _make_layer(self, block, out_channels, blocks, expansion):
        layers = []
        for _ in range(blocks):
            layers.append(block(out_channels, expansion=expansion))
        return nn.Sequential(*layers)

    def forward(self, x, skip_connections):
        deep_supervision_outputs = []
        
        # Decoder stages
        for i in range(4):
            skip = skip_connections[i]
            
            # Pass through MedNeXt block and MedNeXtUp
            x = self.decoder_stages[i](x)
            
            # Concatenate with skip connection
            x = torch.cat([x, skip], dim=1)
            
            # 1x1 convolution to reduce channels from 2 * skip_channels[i] to skip_channels[i]
            x = self.conv1x1_layers[i](x)
            
            # Apply Deep Supervision
            deep_supervision_output = self.deep_supervision_layers[i](x)
            deep_supervision_outputs.append(deep_supervision_output)
        
        # Final segmentation head stage
        final_output = self.decoder_stages[4](x)
        
        return final_output, deep_supervision_outputs


def analyze_mednext_encoder():
    # 创建 MedNeXtEncoder 实例
    encoder = MedNeXtEncoder(
        in_channels=3,  # 输入通道数
        C=64,  # 基础通道数
        encoder_blocks=[3, 4, 8, 8],  # 每个encoder阶段的MedNeXtBlock堆叠次数
        encoder_expansion=[3, 4, 8, 8]  # 每个encoder阶段的扩展倍数
    ).cuda()

    # 创建测试输入张量
    input_tensor = torch.randn(1, 3, 224, 224).cuda()

    # 使用 torchinfo 打印模型摘要
    print("Model Summary:")
    summary(encoder, input_size=(1, 3, 224, 224))

    # 使用 fvcore 统计参数量和 FLOPs
    flops = FlopCountAnalysis(encoder, input_tensor)
    params = parameter_count_table(encoder)

    print(f"Total FLOPs: {flops.total() / 1e9} GFLOPs")
    print(params)

def analyze_mednext_decoder():
    # 定义解码器参数
    in_channels = 512
    skip_channels = [256, 128, 64, 32]  # 假设skip connections的通道数
    decoder_blocks = [8, 8, 8, 4, 3]  # 每个阶段的MedNeXtBlock堆叠次数
    decoder_expansion = [8, 8, 8, 4, 3]  # 每个阶段的扩展倍数
    num_classes = 9  # 假设输出类别数为10

    # 创建 MedNeXtDecoder 实例
    decoder = MedNeXtDecoder(
        in_channels=in_channels,
        skip_channels=skip_channels,
        decoder_blocks=decoder_blocks,
        decoder_expansion=decoder_expansion,
        num_classes=num_classes
    )

    # 创建测试输入张量 (1, 512, 14, 14)
    input_tensor = torch.randn(1, 512, 14, 14)

    # 模拟 skip connections 输入
    skip_connections = [
        torch.randn(1, skip_channels[0], 28, 28),
        torch.randn(1, skip_channels[1], 56, 56),
        torch.randn(1, skip_channels[2], 112, 112),
        torch.randn(1, skip_channels[3], 224, 224),
    ]

    # 使用 torchinfo 打印模型摘要
    print("Model Summary:")
    summary(decoder, input_data=(input_tensor, skip_connections))

    # 使用 fvcore 统计参数量和 FLOPs
    flops = FlopCountAnalysis(decoder, (input_tensor, skip_connections))
    params = parameter_count_table(decoder)

    print(f"Total FLOPs: {flops.total() / 1e9} GFLOPs")
    print(params)

# 执行分析函数
analyze_mednext_decoder()
analyze_mednext_encoder()