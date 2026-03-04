"""
Enhanced Backbone Module

增强型Backbone，包含：
- CNN编码器（复用Part2架构）
- CBAM注意力模块
- 金字塔池化模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module)

    包含通道注意力和空间注意力
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).flatten(1))
        max_out = self.fc(self.max_pool(x).flatten(1))
        channel_att = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x


class PyramidPoolModule(nn.Module):
    """
    金字塔池化模块 (PSP)

    多尺度特征提取，改善小目标分割
    """
    def __init__(self, in_channels: int, pool_scales: Tuple[int, ...] = (1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList()
        self.pool_channels = in_channels // len(pool_scales)

        for scale in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, self.pool_channels, 1, bias=False),
                    nn.BatchNorm2d(self.pool_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            fused: [B, C, H, W]
        """
        h, w = x.shape[2:]
        pyramids = [x]

        for stage in self.stages:
            pyramid = stage(x)
            pyramids.append(F.interpolate(pyramid, size=(h, w), mode='bilinear', align_corners=False))

        return torch.cat(pyramids, dim=1)


class ConvBlock(nn.Module):
    """卷积块: Conv2d + BN + ReLU"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EnhancedBackbone(nn.Module):
    """
    增强型Backbone

    基于 Part2 的 CNN 架构，添加 CBAM 和 金字塔池化
    """
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [64, 128, 256, 512],
        use_cbam: bool = True,
        use_pyramid_pool: bool = True,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
        cbam_reduction: int = 16
    ):
        super().__init__()

        self.channels = channels
        self.use_cbam = use_cbam
        self.use_pyramid_pool = use_pyramid_pool

        # 编码器: 4个stage
        self.encoder = nn.ModuleList()
        current_channels = in_channels

        for out_ch in channels:
            # 每个stage: ConvBlock -> (可选)CBAM -> 下采样
            self.encoder.append(nn.ModuleDict({
                'conv': ConvBlock(current_channels, out_ch),
                'cbam': CBAM(out_ch, cbam_reduction) if use_cbam else nn.Identity(),
                'pool': nn.MaxPool2d(2) if out_ch < channels[-1] else nn.Identity()
            }))
            current_channels = out_ch

        # 金字塔池化
        if use_pyramid_pool:
            self.pyramid_pool = PyramidPoolModule(channels[-1], pool_scales)
            psp_channels = channels[-1] + (channels[-1] // len(pool_scales) * len(pool_scales))
        else:
            psp_channels = channels[-1]

        self.out_channels = channels.copy()
        self.out_channels[-1] = psp_channels  # 更新最后一层的通道数
        self.final_channels = psp_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: List of [B, C_i, H_i, W_i] for each stage
        """
        features = []

        for stage in self.encoder:
            x = stage['conv'](x)
            x = stage['cbam'](x)
            features.append(x)
            x = stage['pool'](x)

        # 金字塔池化作用于最后一层
        if self.use_pyramid_pool:
            features[-1] = self.pyramid_pool(features[-1])

        return features


class ProjectionHead(nn.Module):
    """特征投影头: 将各层特征投影到统一维度"""
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.projections = nn.ModuleList()
        for in_ch in in_channels_list:
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [proj(feat) for proj, feat in zip(self.projections, features)]


if __name__ == "__main__":
    # 测试Backbone
    model = EnhancedBackbone(
        in_channels=3,
        channels=[64, 128, 256, 512],
        use_cbam=True,
        use_pyramid_pool=True
    )

    x = torch.randn(2, 3, 256, 256)
    features = model(x)

    print("EnhancedBackbone output shapes:")
    for i, feat in enumerate(features):
        print(f"  Stage {i}: {feat.shape}")

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
