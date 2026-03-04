"""
Semantic Match Loss

语义匹配损失：确保模型预测特征与CLIP语义特征一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SemanticMatchLoss(nn.Module):
    """
    语义匹配损失

    使用余弦相似度衡量预测特征与CLIP特征的一致性
    目标：最小化语义偏移率
    """
    def __init__(
        self,
        loss_weight: float = 1.0,
        temperature: float = 0.07
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred_features: [B, C] or [B, C, H, W] 模型预测特征
            target_features: [B, C] CLIP目标特征
            mask: Optional mask for valid pixels
        Returns:
            loss: scalar
        """
        # 处理不同形状的输入
        if pred_features.dim() == 4:  # [B, C, H, W]
            # 全局平均池化
            pred_features = pred_features.mean(dim=[2, 3])  # [B, C]

        # L2归一化
        pred_norm = F.normalize(pred_features, dim=-1)
        target_norm = F.normalize(target_features, dim=-1)

        # 余弦相似度
        cos_sim = (pred_norm * target_norm).sum(dim=-1)  # [B]

        # 温度缩放
        cos_sim = cos_sim / self.temperature

        # 损失：1 - 余弦相似度
        loss = 1 - cos_sim.mean()

        return loss * self.loss_weight


class CLIPGuidedLoss(nn.Module):
    """
    CLIP引导损失

    使用CLIP模型直接计算相似度损失
    """
    def __init__(
        self,
        loss_weight: float = 1.0,
        clip_model_name: str = "openai/clip-vit-base-patch32"
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.clip_model_name = clip_model_name
        self._clip_loaded = False

    def _load_clip(self):
        """延迟加载CLIP"""
        if self._clip_loaded:
            return

        try:
            from transformers import CLIPModel
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name)
            for param in self.clip_model.parameters():
                param.requires_grad = False
            self._clip_loaded = True
        except:
            print("[Warning] CLIP model load failed, using fallback")

    def forward(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_images: [B, 3, H, W] 预测图像
            target_images: [B, 3, H, W] 目标图像
        Returns:
            loss: scalar
        """
        self._load_clip()

        # 调整尺寸到CLIP输入要求
        if pred_images.shape[2] != 224:
            pred_images = F.interpolate(pred_images, size=(224, 224), mode='bilinear')
            target_images = F.interpolate(target_images, size=(224, 224), mode='bilinear')

        if hasattr(self, 'clip_model'):
            with torch.no_grad():
                target_features = self.clip_model.get_image_features(pixel_values=target_images)

            pred_features = self.clip_model.get_image_features(pixel_values=pred_images)

            # 余弦相似度损失
            pred_norm = F.normalize(pred_features, dim=-1)
            target_norm = F.normalize(target_features, dim=-1)
            similarity = (pred_norm * target_norm).sum(dim=-1)
            loss = (1 - similarity).mean()
        else:
            # Fallback: MSE损失
            loss = F.mse_loss(pred_images, target_images)

        return loss * self.loss_weight


class SemanticConsistencyLoss(nn.Module):
    """
    语义一致性损失

    确保不同尺度的特征在语义上保持一致
    """
    def __init__(
        self,
        loss_weight: float = 0.5
    ):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        features_list: list,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features_list: List of [B, C, H, W] 多尺度特征
            target_features: [B, C] 目标语义特征
        Returns:
            loss: scalar
        """
        total_loss = 0.0
        count = 0

        target_expanded = target_features.unsqueeze(-1).unsqueeze(-1)

        for features in features_list:
            # 池化到相同维度
            pooled = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)

            # 余弦相似度
            pred_norm = F.normalize(pooled, dim=-1)
            target_norm = F.normalize(target_expanded.squeeze(-1).squeeze(-1), dim=-1)

            similarity = (pred_norm * target_norm).sum(dim=-1)
            loss = (1 - similarity).mean()

            total_loss += loss
            count += 1

        return (total_loss / count) * self.loss_weight if count > 0 else torch.tensor(0.0)


if __name__ == "__main__":
    # 测试语义匹配损失
    print("Testing Semantic Match Loss...")

    loss_fn = SemanticMatchLoss(loss_weight=1.0)

    # 测试数据
    pred_feat = torch.randn(4, 256, 64, 64)
    target_feat = torch.randn(4, 256)

    loss = loss_fn(pred_feat, target_feat)
    print(f"Loss: {loss.item():.4f}")

    # 测试CLIP引导损失
    print("\nTesting CLIP Guided Loss...")
    clip_loss = CLIPGuidedLoss(loss_weight=0.5)

    pred_img = torch.randn(2, 3, 256, 256)
    target_img = torch.randn(2, 3, 256, 256)

    loss = clip_loss(pred_img, target_img)
    print(f"CLIP Loss: {loss.item():.4f}")

    # 测试语义一致性损失
    print("\nTesting Semantic Consistency Loss...")
    consist_loss = SemanticConsistencyLoss(loss_weight=0.3)

    features = [torch.randn(2, 256, 64, 64), torch.randn(2, 256, 32, 32)]
    target = torch.randn(2, 256)

    loss = consist_loss(features, target)
    print(f"Consistency Loss: {loss.item():.4f}")
