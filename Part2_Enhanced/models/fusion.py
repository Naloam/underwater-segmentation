"""
Lightweight Fusion Module

轻量级特征融合模块
使用多头注意力融合视觉、CLIP语义和扩散特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class LightweightFusion(nn.Module):
    """
    轻量级特征融合模块

    使用多头注意力融合三种特征源：
    - 视觉特征 (Backbone)
    - CLIP语义特征
    - 扩散特征
    """
    def __init__(
        self,
        visual_dim: int = 256,
        semantic_dim: int = 256,
        diffusion_dim: int = 256,
        out_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.semantic_dim = semantic_dim
        self.diffusion_dim = diffusion_dim
        self.out_dim = out_dim

        # 维度对齐投影
        self.visual_proj = nn.Conv2d(visual_dim, out_dim, 1) if visual_dim != out_dim else nn.Identity()
        self.semantic_proj = nn.Linear(semantic_dim, out_dim)
        self.diffusion_proj = nn.Conv2d(diffusion_dim, out_dim, 1) if diffusion_dim != out_dim else nn.Identity()

        # 多头注意力融合
        self.cross_attn_v = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_d = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)

        # 层归一化
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(out_dim)

        # 融合权重
        self.alpha_v = nn.Parameter(torch.ones(1))
        self.alpha_s = nn.Parameter(torch.ones(1))
        self.alpha_d = nn.Parameter(torch.ones(1))

    def forward(
        self,
        visual_feat: torch.Tensor,
        semantic_feat: torch.Tensor,
        diffusion_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_feat: [B, C_v, H, W] 视觉特征
            semantic_feat: [B, C_s] CLIP语义特征
            diffusion_feat: [B, C_d, H, W] 扩散特征
        Returns:
            fused: [B, out_dim, H, W] 融合特征
        """
        B, _, H, W = visual_feat.shape

        # 维度对齐
        visual = self.visual_proj(visual_feat)  # [B, out_dim, H, W]
        semantic = self.semantic_proj(semantic_feat)  # [B, out_dim]
        diffusion = self.diffusion_proj(diffusion_feat)  # [B, out_dim, H, W]

        # 展平视觉特征用于注意力
        visual_flat = visual.flatten(2).permute(0, 2, 1)  # [B, H*W, out_dim]
        diffusion_flat = diffusion.flatten(2).permute(0, 2, 1)  # [B, H*W, out_dim]

        # 扩展语义特征
        semantic_expanded = semantic.unsqueeze(1).expand(-1, H * W, -1)  # [B, H*W, out_dim]

        # 交叉注意力: Visual <- Semantic
        v_attn, _ = self.cross_attn_v(
            visual_flat,  # query
            semantic_expanded,  # key
            semantic_expanded  # value
        )
        v_attn = self.norm1(visual_flat + v_attn)

        # 交叉注意力: Visual <- Diffusion
        vd_attn, _ = self.cross_attn_d(
            v_attn,  # query
            diffusion_flat,  # key
            diffusion_flat  # value
        )
        vd_attn = self.norm2(v_attn + vd_attn)

        # 前馈网络
        ffn_out = self.ffn(vd_attn)
        fused_flat = self.norm3(vd_attn + ffn_out)

        # 重塑回空间维度
        fused = fused_flat.permute(0, 2, 1).reshape(B, self.out_dim, H, W)

        return fused


class MultiscaleFusion(nn.Module):
    """
    多尺度特征融合

    融合不同尺度的特征（来自Backbone的多层输出）
    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_dim: int = 256,
        num_heads: int = 4,
        semantic_dim: int = 256,
        diffusion_dim: int = 256
    ):
        super().__init__()

        self.num_scales = len(in_channels_list)
        self.out_dim = out_dim

        # 每个尺度的融合模块
        self.fusion_modules = nn.ModuleList()
        for i, in_ch in enumerate(in_channels_list):
            # 为每个尺度计算输出尺寸
            scale_factor = 2 ** (self.num_scales - 1 - i)
            self.fusion_modules.append(
                ScaleFusionModule(
                    visual_dim=in_ch,
                    semantic_dim=semantic_dim,
                    diffusion_dim=diffusion_dim,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    scale_factor=scale_factor
                )
            )

    def forward(
        self,
        visual_features: List[torch.Tensor],
        semantic_feat: torch.Tensor,
        diffusion_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            visual_features: List of [B, C_i, H_i, W_i]
            semantic_feat: [B, C_s]
            diffusion_features: List of [B, C_d, H_i, W_i]
        Returns:
            fused_features: List of [B, out_dim, H_i, W_i]
        """
        fused = []
        for i, (v, d) in enumerate(zip(visual_features, diffusion_features)):
            fused.append(self.fusion_modules[i](v, semantic_feat, d))

        return fused


class ScaleFusionModule(nn.Module):
    """单尺度融合模块"""
    def __init__(
        self,
        visual_dim: int,
        semantic_dim: int,
        diffusion_dim: int,
        out_dim: int,
        num_heads: int,
        scale_factor: int
    ):
        super().__init__()

        self.scale_factor = scale_factor
        self.out_dim = out_dim

        # 上采样CLIP特征到对应尺度
        self.semantic_upsample = nn.Sequential(
            nn.Linear(semantic_dim, out_dim * scale_factor * scale_factor),
            nn.GELU()
        )

        # 轻量级融合
        self.fusion = nn.Sequential(
            nn.Conv2d(visual_dim + diffusion_dim, out_dim, 1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_dim),
            nn.GELU()
        )

        # 语义门控
        self.gate = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(out_dim // 4, out_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        visual_feat: torch.Tensor,
        semantic_feat: torch.Tensor,
        diffusion_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            visual_feat: [B, C_v, H, W]
            semantic_feat: [B, C_s]
            diffusion_feat: [B, C_d, H, W]
        Returns:
            fused: [B, out_dim, H, W]
        """
        B, _, H, W = visual_feat.shape

        # 融合视觉和扩散特征
        combined = torch.cat([visual_feat, diffusion_feat], dim=1)
        base_feat = self.fusion(combined)

        # 上采样语义特征 - 先reshape到scale_factor x scale_factor，再插值到H x W
        semantic_map = self.semantic_upsample(semantic_feat)  # [B, out_dim * scale_factor * scale_factor]
        semantic_map = semantic_map.reshape(B, self.out_dim, self.scale_factor, self.scale_factor)
        # 插值到目标尺寸
        semantic_map = torch.nn.functional.interpolate(semantic_map, size=(H, W), mode='bilinear', align_corners=False)

        # 语义门控
        gate = self.gate(base_feat)
        fused = base_feat * gate + semantic_map * (1 - gate)

        return fused


class FeatureFusionNeck(nn.Module):
    """
    特征融合颈部网络

    整合所有特征融合逻辑
    """
    def __init__(
        self,
        in_channels_list: List[int],
        semantic_dim: int = 256,
        diffusion_dim: int = 256,
        out_dim: int = 256,
        num_heads: int = 4,
        use_clip: bool = True,
        use_diffusion: bool = True
    ):
        super().__init__()

        self.use_clip = use_clip
        self.use_diffusion = use_diffusion
        self.semantic_dim = semantic_dim
        self.diffusion_dim = diffusion_dim
        self.out_dim = out_dim

        # 维度投影
        self.visual_projs = nn.ModuleList([
            nn.Conv2d(in_ch, out_dim, 1) for in_ch in in_channels_list
        ])

        # 多尺度融合
        if use_clip or use_diffusion:
            self.multiscale_fusion = MultiscaleFusion(
                [out_dim] * len(in_channels_list),
                out_dim,
                num_heads,
                semantic_dim if use_clip else out_dim,
                diffusion_dim if use_diffusion else out_dim
            )

    def forward(
        self,
        visual_features: List[torch.Tensor],
        semantic_feat: Optional[torch.Tensor] = None,
        diffusion_features: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        Args:
            visual_features: List of [B, C_i, H_i, W_i]
            semantic_feat: [B, C_s] or None
            diffusion_features: List of [B, C_d, H_i, W_i] or None
        Returns:
            fused_features: List of [B, out_dim, H_i, W_i]
        """
        # 投影视觉特征
        visual_proj = [proj(v) for proj, v in zip(self.visual_projs, visual_features)]

        # 如果不使用额外特征，直接返回
        if not self.use_clip and not self.use_diffusion:
            return visual_proj

        # 准备语义和扩散特征
        semantic = semantic_feat if self.use_clip else torch.zeros(
            visual_proj[0].shape[0], self.semantic_dim,
            device=visual_proj[0].device
        )

        diffusion = diffusion_features if self.use_diffusion else [
            torch.zeros_like(v) for v in visual_proj
        ]

        # 多尺度融合
        fused = self.multiscale_fusion(visual_proj, semantic, diffusion)

        return fused


if __name__ == "__main__":
    # 测试融合模块
    print("Testing Lightweight Fusion...")

    # 单尺度融合测试
    fusion = LightweightFusion(
        visual_dim=256,
        semantic_dim=256,
        diffusion_dim=256,
        out_dim=256,
        num_heads=4
    )

    visual = torch.randn(2, 256, 64, 64)
    semantic = torch.randn(2, 256)
    diffusion = torch.randn(2, 256, 64, 64)

    fused = fusion(visual, semantic, diffusion)
    print(f"Fused shape: {fused.shape}")

    # 多尺度融合测试
    print("\nTesting Multiscale Fusion...")
    neck = FeatureFusionNeck(
        in_channels_list=[64, 128, 256, 512],
        semantic_dim=256,
        diffusion_dim=256,
        out_dim=256,
        num_heads=4,
        use_clip=True,
        use_diffusion=True
    )

    visual_feats = [torch.randn(2, 64, 128, 128), torch.randn(2, 128, 64, 64),
                    torch.randn(2, 256, 32, 32), torch.randn(2, 512, 16, 16)]
    semantic = torch.randn(2, 256)
    diffusion_feats = [torch.randn(2, 256, 128, 128), torch.randn(2, 256, 64, 64),
                       torch.randn(2, 256, 32, 32), torch.randn(2, 256, 16, 16)]

    fused_feats = neck(visual_feats, semantic, diffusion_feats)
    print("Multiscale fused shapes:")
    for i, feat in enumerate(fused_feats):
        print(f"  Scale {i}: {feat.shape}")

    # 参数量
    total_params = sum(p.numel() for p in neck.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
