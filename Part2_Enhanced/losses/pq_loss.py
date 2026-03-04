"""
Panoptic Quality Loss

PQ (Panoptic Quality) 损失：全景分割专属损失
结合分割质量(SQ)和区域质量(RQ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PQLoss(nn.Module):
    """
    简化版PQ损失

    PQ = SQ × RQ
    - SQ (Segmentation Quality): 分割质量，基于IoU
    - RQ (Recognition Quality): 区域质量，基于匹配
    """
    def __init__(
        self,
        num_classes: int = 8,
        loss_weight: float = 2.0,
        iou_threshold: float = 0.5,
        ignore_index: int = -1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.iou_threshold = iou_threshold
        self.ignore_index = ignore_index

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: [B, num_classes, H, W] 预测logits
            target_masks: [B, H, W] 目标掩码 (类别标签)
        Returns:
            loss: scalar PQ损失
        """
        # Softmax预测
        pred_probs = F.softmax(pred_logits, dim=1)  # [B, C, H, W]
        pred_labels = torch.argmax(pred_probs, dim=1)  # [B, H, W]

        # 计算每个类别的IoU
        iou_per_class = self._compute_iou_per_class(pred_labels, target_masks)

        # 计算PQ
        pq_loss = self._compute_pq_loss(iou_per_class, pred_labels, target_masks)

        return pq_loss * self.loss_weight

    def _compute_iou_per_class(
        self,
        pred_labels: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        计算每个类别的IoU

        Args:
            pred_labels: [B, H, W]
            target_masks: [B, H, W]
        Returns:
            iou: [C] 每个类别的IoU
        """
        iou_list = []

        for cls in range(self.num_classes):
            pred_mask = (pred_labels == cls).float()
            target_mask = (target_masks == cls).float()

            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum() - intersection

            if union > 0:
                iou = intersection / union
            else:
                iou = torch.tensor(0.0, device=pred_labels.device)

            iou_list.append(iou)

        return torch.stack(iou_list)

    def _compute_pq_loss(
        self,
        iou_per_class: torch.Tensor,
        pred_labels: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        计算PQ损失

        Args:
            iou_per_class: [C] 每个类别的IoU
            pred_labels: [B, H, W]
            target_masks: [B, H, W]
        Returns:
            pq_loss: scalar
        """
        # 匹配的像素 (IoU > threshold)
        matched = iou_per_class > self.iou_threshold

        # SQ: 匹配类别的平均IoU
        if matched.any():
            sq = iou_per_class[matched].mean()
        else:
            sq = torch.tensor(0.0, device=iou_per_class.device)

        # RQ: 匹配类别的比例
        rq = matched.float().mean()

        # PQ = SQ × RQ
        pq = sq * rq

        # PQ损失 = 1 - PQ
        pq_loss = 1 - pq

        return pq_loss


class FocalLoss(nn.Module):
    """
    Focal Loss

    用于处理类别不平衡
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -1
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: [B, C, H, W]
            target_masks: [B, H, W]
        Returns:
            loss: scalar
        """
        ce_loss = F.cross_entropy(
            pred_logits,
            target_masks,
            ignore_index=self.ignore_index,
            reduction='none'
        )

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss

    用于分割任务，对小目标更敏感
    """
    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = -1
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_logits: [B, C, H, W]
            target_masks: [B, H, W]
        Returns:
            loss: scalar
        """
        pred_probs = F.softmax(pred_logits, dim=1)

        # One-hot编码
        num_classes = pred_logits.shape[1]
        target_one_hot = F.one_hot(target_masks, num_classes).permute(0, 3, 1, 2).float()

        # 计算Dice
        intersection = (pred_probs * target_one_hot).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return (1 - dice).mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数

    结合CE、Focal、Dice、PQ、语义匹配损失
    """
    def __init__(
        self,
        num_classes: int = 8,
        ce_weight: float = 1.0,
        focal_weight: float = 0.5,
        dice_weight: float = 1.0,
        pq_weight: float = 2.0,
        semantic_weight: float = 1.0,
        ignore_index: int = -1
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.pq_weight = pq_weight
        self.semantic_weight = semantic_weight

        # 子损失
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(smooth=1.0, ignore_index=ignore_index)
        self.pq_loss = PQLoss(num_classes, loss_weight=1.0, ignore_index=ignore_index)

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_masks: torch.Tensor,
        semantic_features: Optional[torch.Tensor] = None,
        target_semantic: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred_logits: [B, C, H, W]
            target_masks: [B, H, W]
            semantic_features: Optional [B, C] 模型语义特征
            target_semantic: Optional [B, C] CLIP目标特征
        Returns:
            total_loss: scalar
            loss_dict: 各项损失的字典
        """
        loss_dict = {}

        # CE Loss
        ce_loss = F.cross_entropy(pred_logits, target_masks, ignore_index=-1)
        loss_dict['ce'] = ce_loss.item()

        # Focal Loss
        if self.focal_weight > 0:
            focal_loss = self.focal_loss(pred_logits, target_masks)
            loss_dict['focal'] = focal_loss.item()
        else:
            focal_loss = torch.tensor(0.0, device=pred_logits.device)

        # Dice Loss
        if self.dice_weight > 0:
            dice_loss = self.dice_loss(pred_logits, target_masks)
            loss_dict['dice'] = dice_loss.item()
        else:
            dice_loss = torch.tensor(0.0, device=pred_logits.device)

        # PQ Loss
        if self.pq_weight > 0:
            pq_loss = self.pq_loss(pred_logits, target_masks)
            loss_dict['pq'] = pq_loss.item()
        else:
            pq_loss = torch.tensor(0.0, device=pred_logits.device)

        # 语义匹配损失
        if self.semantic_weight > 0 and semantic_features is not None and target_semantic is not None:
            from .semantic_loss import SemanticMatchLoss
            semantic_loss_fn = SemanticMatchLoss(loss_weight=1.0)
            semantic_loss = semantic_loss_fn(semantic_features, target_semantic)
            loss_dict['semantic'] = semantic_loss.item()
        else:
            semantic_loss = torch.tensor(0.0, device=pred_logits.device)

        # 总损失
        total_loss = (
            self.ce_weight * ce_loss +
            self.focal_weight * focal_loss +
            self.dice_weight * dice_loss +
            self.pq_weight * pq_loss +
            self.semantic_weight * semantic_loss
        )

        return total_loss, loss_dict


if __name__ == "__main__":
    # 测试组合损失
    print("Testing Combined Loss...")

    loss_fn = CombinedLoss(
        num_classes=8,
        ce_weight=1.0,
        focal_weight=0.5,
        dice_weight=1.0,
        pq_weight=2.0,
        semantic_weight=1.0
    )

    # 测试数据
    pred_logits = torch.randn(2, 8, 256, 256)
    target_masks = torch.randint(0, 8, (2, 256, 256))
    semantic_feat = torch.randn(2, 256)
    target_semantic = torch.randn(2, 256)

    total_loss, loss_dict = loss_fn(pred_logits, target_masks, semantic_feat, target_semantic)

    print(f"Total Loss: {total_loss.item():.4f}")
    print("Loss breakdown:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
