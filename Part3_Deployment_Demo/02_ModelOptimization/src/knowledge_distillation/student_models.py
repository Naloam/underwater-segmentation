"""
轻量级学生模型定义

用于知识蒸馏的轻量级分割模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any


class MobileNetV3Encoder(nn.Module):
    """
    MobileNetV3编码器作为轻量级backbone
    """

    def __init__(self, in_channels: int = 3, width_mult: float = 0.5):
        super().__init__()

        # 初始卷积
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )

        # MobileNetV3 inverted residual blocks (简化版)
        self.layers = nn.ModuleList([
            # Stage 1: 16 -> 24
            self._make_layer(16, 24, 2, width_mult),
            # Stage 2: 24 -> 40
            self._make_layer(24, 40, 2, width_mult),
            # Stage 3: 40 -> 48
            self._make_layer(40, 48, 1, width_mult),
            # Stage 4: 48 -> 96
            self._make_layer(48, 96, 2, width_mult),
        ])

        self.out_channels = [24, 40, 48, 96]

    def _make_layer(self, in_ch: int, out_ch: int, stride: int, width_mult: float):
        """创建一层inverted residual"""
        mid_ch = int(in_ch * width_mult * 6)
        return nn.Sequential(
            # 1x1 expansion
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            # 3x3 depthwise
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            # 1x1 projection
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向传播，返回多尺度特征"""
        x = self.conv_bn(x)

        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return features


class LightSegmentationHead(nn.Module):
    """
    轻量级分割头
    """

    def __init__(self, in_channels: List[int], num_classes: int):
        super().__init__()

        # 特征融合
        self.fusion = nn.ModuleList([
            nn.Conv2d(ch, 64, 1) for ch in in_channels
        ])

        # 上采样和融合
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            for _ in range(len(in_channels))
        ])

        # 最终分类
        total_channels = 64 * len(in_channels)
        self.classifier = nn.Sequential(
            nn.Conv2d(total_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 多尺度特征列表 [f1, f2, f3, f4]
        Returns:
            分割logits
        """
        # 特征融合
        fused_features = []
        for i, (feat, fusion_conv) in enumerate(zip(features, self.fusion)):
            x = fusion_conv(feat)
            # 上采样到同一尺寸
            target_size = features[0].shape[2:]
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            fused_features.append(x)

        # 拼接
        x = torch.cat(fused_features, dim=1)

        # 分类
        x = self.classifier(x)

        # 上采样到输入尺寸
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x


class LightweightSegmentor(nn.Module):
    """
    轻量级分割模型

    基于MobileNetV3的轻量级全景分割模型。
    """

    def __init__(self, num_classes: int = 8, width_mult: float = 0.5):
        super().__init__()

        self.backbone = MobileNetV3Encoder(width_mult=width_mult)
        self.head = LightSegmentationHead(
            self.backbone.out_channels,
            num_classes
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.backbone(x)
        output = self.head(features)
        return output

    def get_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "name": "LightweightSegmentor",
            "num_classes": self.num_classes,
            "params": total_params / 1e6,  # M
            "backbone": "MobileNetV3"
        }


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失

    包含特征蒸馏和输出蒸馏。
    """

    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 4.0,
        feature_weight: float = 1.0
    ):
        super().__init__()
        self.alpha = alpha  # 软标签权重
        self.temperature = temperature
        self.feature_weight = feature_weight

    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_features: List[torch.Tensor] = None,
        teacher_features: List[torch.Tensor] = None,
        target: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算蒸馏损失

        Args:
            student_output: 学生模型输出
            teacher_output: 教师模型输出
            student_features: 学生模型特征
            teacher_features: 教师模型特征
            target: 真实标签

        Returns:
            总损失, 损失详情字典
        """
        losses = {}

        # KL散度损失（软标签）
        soft_loss = F.kl_div(
            F.log_softmax(student_output / self.temperature, dim=1),
            F.softmax(teacher_output / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)

        # 硬标签损失
        if target is not None:
            hard_loss = F.cross_entropy(student_output, target)
            losses['hard_loss'] = hard_loss.item()
            total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = soft_loss

        losses['soft_loss'] = soft_loss.item()

        # 特征蒸馏损失（可选）
        if student_features is not None and teacher_features is not None:
            feature_loss = 0
            for sf, tf in zip(student_features, teacher_features):
                # 调整尺寸
                if sf.shape != tf.shape:
                    tf = F.interpolate(tf, size=sf.shape[2:], mode='bilinear')

                feature_loss += F.mse_loss(sf, tf)

            feature_loss = feature_loss / len(student_features)
            total_loss = total_loss + self.feature_weight * feature_loss
            losses['feature_loss'] = feature_loss.item()

        return total_loss, losses


# 预定义的模型配置
STUDENT_MODEL_CONFIGS = {
    'mobilenet_small': {
        'num_classes': 8,
        'width_mult': 0.35,
        'target_params': 3,  # M
    },
    'mobilenet_base': {
        'num_classes': 8,
        'width_mult': 0.5,
        'target_params': 5,  # M
    },
    'mobilenet_large': {
        'num_classes': 8,
        'width_mult': 0.75,
        'target_params': 8,  # M
    }
}


def create_student_model(config_name: str = 'mobilenet_base') -> LightweightSegmentor:
    """
    创建学生模型

    Args:
        config_name: 配置名称

    Returns:
        学生模型实例
    """
    config = STUDENT_MODEL_CONFIGS.get(config_name, STUDENT_MODEL_CONFIGS['mobilenet_base'])
    return LightweightSegmentor(**config)


if __name__ == '__main__':
    # 测试模型
    model = create_student_model('mobilenet_base')
    info = model.get_info()
    print(f"模型信息: {info}")

    # 测试前向传播
    x = torch.randn(1, 3, 512, 512)
    output = model(x)
    print(f"输出形状: {output.shape}")
