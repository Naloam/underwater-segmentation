
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import Config
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.utils import mask2array


# ======================== 自定义核心模块 ========================
class FeatureFusionNeck(nn.Module):
    """语义-视觉双特征融合模块"""
    def __init__(self, in_channels, out_channels, num_outs, fusion_type='concat_attn'):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_ch in in_channels:
            self.convs.append(
                ConvModule(in_ch, out_channels, 1, norm_cfg=dict(type='GN', num_groups=32), act_cfg=dict(type='ReLU'))
            )
        self.attn_fusion = nn.MultiheadAttention(out_channels, num_heads=8, batch_first=True)

    def forward(self, inputs):
        feats = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        fused_feats = []
        for feat in feats:
            B, C, H, W = feat.shape
            feat_flat = feat.flatten(2).permute(0, 2, 1)
            feat_fused, _ = self.attn_fusion(feat_flat, feat_flat, feat_flat)
            feat_fused = feat_fused.permute(0, 2, 1).reshape(B, C, H, W)
            fused_feats.append(feat_fused)
        return fused_feats


class CBAM(nn.Module):
    """CBAM 注意力模块（通道+空间）"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C = x.shape[:2]
        channel_att = torch.sigmoid(
            self.fc(self.avg_pool(x).view(B, C)) + self.fc(self.max_pool(x).view(B, C))
        ).view(B, C, 1, 1)
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_att


# ======================== Mask2Former 完整模型 ========================
@SEGMENTORS.register_module()
class Mask2Former(BaseSegmentor):
    """Mask2Former 全景/语义分割模型（适配SUIM 8分类）"""
    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)

        # 主干网络 Swin Transformer
        self.backbone = builder.build_backbone(backbone)

        # 自定义特征融合 Neck
        self.neck = builder.build_neck(neck) if neck else None

        # Mask2Former 预测头
        panoptic_head.update(train_cfg=train_cfg)
        panoptic_head.update(test_cfg=test_cfg)
        self.panoptic_head = builder.build_head(panoptic_head)

        # 初始化权重
        self.init_weights(pretrained=pretrained)

    def extract_feat(self, img):
        """提取图像特征"""
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """训练前向传播"""
        x = self.extract_feat(img)
        losses = self.panoptic_head.forward_train(x, img_metas, gt_semantic_seg)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        """测试/推理"""
        x = self.extract_feat(img)
        seg_logits = self.panoptic_head.forward_test(x, img_metas, **kwargs)
        return seg_logits

    def forward(self, img, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)


# ======================== 模型配置（直接实例化） ========================
def build_mask2former_suim(num_classes=8, device='cuda'):
    """
    构建用于 SUIM 水下分割的 Mask2Former 模型
    :return: 完整模型
    """
    model_cfg = dict(
        type='Mask2Former',
        backbone=dict(
            type='SwinTransformer',
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            mlp_ratio=4,
            out_indices=(0, 1, 2, 3)
        ),
        neck=dict(
            type='FeatureFusionNeck',
            in_channels=[128, 256, 512, 1024],
            out_channels=256,
            num_outs=4
        ),
        panoptic_head=dict(
            type='Mask2FormerHead',
            in_channels=[256, 256, 256, 256],
            feat_channels=256,
            out_channels=256,
            num_classes=num_classes,
            num_queries=100,
            loss_cls=dict(type='CrossEntropyLoss', loss_weight=2.0),
            loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=5.0),
            loss_dice=dict(type='DiceLoss', loss_weight=5.0),
        ),
        test_cfg=dict(mode='whole')
    )

    model = Mask2Former(**model_cfg)
    return model.to(device)


# ======================== 测试模型是否可用 ========================
if __name__ == '__main__':
    # 测试：创建模型 + 随机输入推理
    model = build_mask2former_suim(num_classes=8)
    dummy_input = torch.randn(1, 3, 512, 512).cuda()  # [B, C, H, W]
    model.eval()

    with torch.no_grad():
        output = model(dummy_input, return_loss=False)

    print("✅ Mask2Former 模型创建成功！")
    print(f"📌 输入尺寸: {dummy_input.shape}")
    print(f"📌 输出尺寸: {output.shape}")