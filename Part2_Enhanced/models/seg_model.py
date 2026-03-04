"""
Segmentation Model

完整的全景分割模型，整合所有模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.model_config import ModelConfig
from .backbone import EnhancedBackbone, ProjectionHead
from .clip_branch import CLIPSemanticBranch
from .diffusion_branch import DiffusionFeatureBranch
from .fusion import FeatureFusionNeck


class SegmentationDecoder(nn.Module):
    """
    分割解码器

    将融合后的多尺度特征解码为分割掩码
    """
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 8
    ):
        super().__init__()

        self.num_classes = num_classes

        # FPN上采样
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for _ in range(4):  # 4个尺度
            self.lateral_convs.append(
                nn.Conv2d(in_channels, in_channels, 1)
            )
            self.output_convs.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1)
            )

        # 最终分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels // 2, num_classes, 1)
        )

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: tuple
    ) -> torch.Tensor:
        """
        Args:
            features: List of [B, C, H_i, W_i]
            target_size: (H, W) 输出尺寸
        Returns:
            logits: [B, num_classes, H, W]
        """
        # FPN风格上采样
        x = features[-1]  # 从最高层开始

        outputs = []
        for i in range(len(features) - 2, -1, -1):
            # 上采样
            x = F.interpolate(x, size=features[i].shape[2:], mode='bilinear', align_corners=False)
            # 侧边连接
            lateral = self.lateral_convs[i](features[i])
            x = x + lateral
            x = self.output_convs[i](x)
            outputs.append(x)

        # 使用最高分辨率的特征
        x = outputs[-1] if outputs else x

        # 分类
        logits = self.classifier(x)

        # 上采样到目标尺寸
        logits = F.interpolate(logits, size=target_size, mode='bilinear', align_corners=False)

        return logits


class SegmentationModel(nn.Module):
    """
    完整的全景分割模型

    整合Backbone、CLIP分支、扩散分支和融合模块
    """
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Backbone
        self.backbone = EnhancedBackbone(
            in_channels=3,
            channels=config.backbone_channels,
            use_cbam=config.use_cbam,
            use_pyramid_pool=config.use_pyramid_pool,
            pool_scales=config.pool_scales,
            cbam_reduction=config.cbam_reduction
        )

        # 投影到统一维度
        self.backbone_proj = ProjectionHead(
            self.backbone.out_channels,
            out_channels=config.fusion_dim
        )

        # CLIP语义分支
        if config.use_clip:
            self.clip_branch = CLIPSemanticBranch(
                model_name=config.clip_model_name,
                embed_dim=config.clip_embed_dim,
                out_dim=config.clip_out_dim,
                freeze=config.freeze_clip
            )
        else:
            self.clip_branch = None

        # 扩散特征分支
        if config.use_diffusion:
            self.diffusion_branch = DiffusionFeatureBranch(
                in_channels=3,
                channels=config.diffusion_channels,
                out_dim=config.diffusion_out_dim
            )
        else:
            self.diffusion_branch = None

        # 特征融合
        self.fusion_neck = FeatureFusionNeck(
            in_channels_list=[config.fusion_dim] * len(config.backbone_channels),
            semantic_dim=config.clip_out_dim if config.use_clip else config.fusion_dim,
            diffusion_dim=config.diffusion_out_dim if config.use_diffusion else config.fusion_dim,
            out_dim=config.fusion_dim,
            num_heads=config.num_heads,
            use_clip=config.use_clip,
            use_diffusion=config.use_diffusion
        )

        # 解码器
        self.decoder = SegmentationDecoder(
            in_channels=config.fusion_dim,
            num_classes=config.num_classes
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] 输入图像
            return_features: 是否返回中间特征
        Returns:
            logits: [B, num_classes, H, W] 分割logits
            features: (可选) 中间特征字典
        """
        input_size = x.shape[2:]

        # Backbone特征提取
        visual_features = self.backbone(x)
        visual_features = self.backbone_proj(visual_features)

        # CLIP语义特征
        semantic_feat = None
        if self.clip_branch is not None:
            # CLIP需要224x224输入
            x_clip = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            semantic_feat = self.clip_branch(x_clip)

        # 扩散特征
        diffusion_features = None
        if self.diffusion_branch is not None:
            diffusion_features = self.diffusion_branch(x)

        # 特征融合
        fused_features = self.fusion_neck(visual_features, semantic_feat, diffusion_features)

        # 解码
        logits = self.decoder(fused_features, input_size)

        if return_features:
            features_dict = {
                'visual': visual_features,
                'semantic': semantic_feat,
                'diffusion': diffusion_features,
                'fused': fused_features
            }
            return logits, features_dict

        return logits

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info = {
            'name': 'Enhanced Segmentation Model',
            'num_classes': self.config.num_classes,
            'input_size': self.config.input_size,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'use_clip': self.config.use_clip,
            'use_diffusion': self.config.use_diffusion,
            'backbone_channels': self.config.backbone_channels,
            'fusion_dim': self.config.fusion_dim
        }

        return info


def create_model(config: ModelConfig = None) -> SegmentationModel:
    """工厂函数：创建分割模型"""
    if config is None:
        from configs.model_config import model_cfg
        config = model_cfg

    model = SegmentationModel(config)
    return model


if __name__ == "__main__":
    # 测试完整模型
    print("Testing Enhanced Segmentation Model...")

    config = ModelConfig(
        num_classes=8,
        input_size=(256, 256),
        backbone_channels=[64, 128, 256, 512],
        use_cbam=True,
        use_clip=True,
        use_diffusion=True
    )

    model = create_model(config)

    # 模拟输入
    x = torch.randn(2, 3, 256, 256)

    print("\nForward pass...")
    logits, features = model(x, return_features=True)

    print(f"Output logits shape: {logits.shape}")
    print(f"\nIntermediate features:")
    print(f"  Visual: {[f.shape for f in features['visual']]}")
    print(f"  Semantic: {features['semantic'].shape if features['semantic'] is not None else 'None'}")
    print(f"  Diffusion: {[f.shape for f in features['diffusion']]} if features['diffusion'] else 'None'")
    print(f"  Fused: {[f.shape for f in features['fused']]}")

    # 模型信息
    info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Name: {info['name']}")
    print(f"  Total params: {info['total_params']:,} ({info['total_params']/1e6:.2f}M)")
    print(f"  Trainable params: {info['trainable_params']:,} ({info['trainable_params']/1e6:.2f}M)")
    print(f"  Frozen params: {info['frozen_params']:,} ({info['frozen_params']/1e6:.2f}M)")
    print(f"  Use CLIP: {info['use_clip']}")
    print(f"  Use Diffusion: {info['use_diffusion']}")

    # 测试推理
    print("\nTesting inference...")
    with torch.no_grad():
        logits = model(x)
    print(f"Inference output shape: {logits.shape}")
    print(f"Predicted classes: {torch.argmax(logits, dim=1).unique()}")
