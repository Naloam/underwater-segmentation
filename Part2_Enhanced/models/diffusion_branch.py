"""
Diffusion Feature Branch Module

简化版扩散特征提取器
使用U-Net编码器提取多尺度特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class UNetEncoder(nn.Module):
    """
    U-Net编码器

    提取多尺度特征，不包含完整的扩散模型
    """
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [64, 128, 256, 512],
        use_attention: bool = False
    ):
        super().__init__()

        self.channels = channels
        self.use_attention = use_attention

        # 编码器各层
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        current_ch = in_channels
        for out_ch in channels:
            # 每层: Conv -> Conv -> (可选)Attention
            block = []
            block.append(nn.Conv2d(current_ch, out_ch, 3, padding=1))
            block.append(nn.GroupNorm(32, out_ch))
            block.append(nn.SiLU())

            block.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
            block.append(nn.GroupNorm(32, out_ch))
            block.append(nn.SiLU())

            if use_attention and out_ch >= 256:
                block.append(SelfAttention(out_ch))

            self.encoder_blocks.append(nn.Sequential(*block))
            self.pools.append(nn.MaxPool2d(2))

            current_ch = out_ch

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: List of [B, C_i, H_i, W_i]
        """
        features = []

        for i, (block, pool) in enumerate(zip(self.encoder_blocks, self.pools)):
            x = block(x)
            features.append(x)
            if i < len(self.encoder_blocks) - 1:  # 最后一层不下采样
                x = pool(x)

        return features


class SelfAttention(nn.Module):
    """自注意力模块"""
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.flatten(2).permute(0, 2, 1)  # [B, H*W, C]

        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)

        return x + attn_out


class DiffusionFeatureBranch(nn.Module):
    """
    扩散特征分支

    使用简化U-Net编码器提取多尺度特征
    不包含完整的扩散采样过程
    """
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [64, 128, 256, 512],
        out_dim: int = 256,
        use_attention: bool = False
    ):
        super().__init__()

        self.channels = channels
        self.out_dim = out_dim

        # U-Net编码器
        self.encoder = UNetEncoder(in_channels, channels, use_attention)

        # 特征投影到统一维度
        self.projections = nn.ModuleList()
        for ch in channels:
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_dim, 1, bias=False),
                    nn.GroupNorm(32, out_dim),
                    nn.SiLU()
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: List of [B, out_dim, H_i, W_i]
        """
        # 提取多尺度特征
        features = self.encoder(x)

        # 投影到统一维度
        projected_features = []
        for feat, proj in zip(features, self.projections):
            projected_features.append(proj(feat))

        return projected_features


class DiffusionTimeEmbedding(nn.Module):
    """
    扩散时间步嵌入（可选）

    如果需要完整的扩散模型，可以添加时间步嵌入
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        # 正弦位置编码
        self.frequencies = embed_dim // 2

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B,] 时间步
        Returns:
            embedding: [B, embed_dim]
        """
        device = t.device
        half = self.frequencies
        freqs = torch.arange(half, device=device, dtype=torch.float32) / half
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


if __name__ == "__main__":
    # 测试扩散特征分支
    print("Testing Diffusion Feature Branch...")

    model = DiffusionFeatureBranch(
        in_channels=3,
        channels=[64, 128, 256, 512],
        out_dim=256,
        use_attention=True
    )

    # 模拟输入
    x = torch.randn(2, 3, 256, 256)

    print("\nForward pass...")
    features = model(x)

    print("Multi-scale feature shapes:")
    for i, feat in enumerate(features):
        print(f"  Scale {i}: {feat.shape}")

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # 测试时间嵌入
    print("\n" + "="*50)
    print("Testing Diffusion Time Embedding...")
    time_embed = DiffusionTimeEmbedding(embed_dim=256)
    t = torch.tensor([0, 50, 100, 500])
    embedding = time_embed(t)
    print(f"Time step shape: {t.shape}")
    print(f"Embedding shape: {embedding.shape}")
