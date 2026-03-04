"""
Evaluation Metrics

评估指标计算模块
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> List[float]:
    """
    计算每个类别的IoU

    Args:
        pred: [H, W] or [B, H, W] 预测标签
        target: [H, W] or [B, H, W] 目标标签
        num_classes: 类别数
    Returns:
        iou_per_class: List[float] 每个类别的IoU
    """
    if pred.dim() == 3:
        pred = pred.flatten()
        target = target.flatten()
    else:
        pred = pred.flatten()
        target = target.flatten()

    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(0.0)

        ious.append(iou.item())

    return ious


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    计算平均IoU

    Args:
        preds: [B, H, W] 预测标签
        targets: [B, H, W] 目标标签
        num_classes: 类别数
    Returns:
        miou: float 平均IoU
    """
    total_iou = 0.0
    count = 0

    for pred, target in zip(preds, targets):
        ious = compute_iou(pred, target, num_classes)
        # 只计算存在的类别
        valid_iou = [iou for iou, i in zip(ious, range(num_classes)) if i in target]
        if valid_iou:
            total_iou += sum(valid_iou) / len(valid_iou)
            count += 1

    return total_iou / count if count > 0 else 0.0


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算像素准确率

    Args:
        preds: [B, H, W] 预测标签
        targets: [B, H, W] 目标标签
    Returns:
        accuracy: float 准确率
    """
    correct = (preds == targets).sum().item()
    total = preds.numel()
    return correct / total


def compute_f1_score(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    计算F1分数

    Args:
        preds: [B, H, W] 预测标签
        targets: [B, H, W] 目标标签
        num_classes: 类别数
    Returns:
        f1: float F1分数
    """
    pred_flat = preds.flatten()
    target_flat = targets.flatten()

    f1_scores = []
    for cls in range(num_classes):
        tp = ((pred_flat == cls) & (target_flat == cls)).sum().item()
        fp = ((pred_flat == cls) & (target_flat != cls)).sum().item()
        fn = ((pred_flat != cls) & (target_flat == cls)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_scores.append(f1)

    return np.mean(f1_scores)


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        preds: [B, H, W] 预测标签
        targets: [B, H, W] 目标标签
        num_classes: 类别数
    Returns:
        metrics: Dict[str, float] 指标字典
    """
    metrics = {
        'miou': compute_miou(preds, targets, num_classes),
        'accuracy': compute_accuracy(preds, targets),
        'f1': compute_f1_score(preds, targets, num_classes)
    }

    # 每个类别的IoU
    ious = []
    for pred, target in zip(preds, targets):
        ious.extend(compute_iou(pred, target, num_classes))

    for i, iou in enumerate(ious[:num_classes]):
        metrics[f'iou_class_{i}'] = iou

    return metrics


class MetricTracker:
    """指标跟踪器"""
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.total_metrics = defaultdict(float)
        self.count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """更新指标"""
        metrics = compute_metrics(preds, targets, self.num_classes)

        for k, v in metrics.items():
            self.total_metrics[k] += v

        self.count += 1

    def get_average(self) -> Dict[str, float]:
        """获取平均指标"""
        return {k: v / self.count for k, v in self.total_metrics.items()}

    def get_summary(self) -> str:
        """获取指标摘要"""
        avg_metrics = self.get_average()

        summary = []
        summary.append(f"mIoU: {avg_metrics['miou']:.4f}")
        summary.append(f"Accuracy: {avg_metrics['accuracy']:.4f}")
        summary.append(f"F1: {avg_metrics['f1']:.4f}")

        return ", ".join(summary)


if __name__ == "__main__":
    # 测试指标计算
    print("Testing Metrics...")

    # 模拟预测和目标
    num_classes = 8
    batch_size = 4
    preds = torch.randint(0, num_classes, (batch_size, 256, 256))
    targets = torch.randint(0, num_classes, (batch_size, 256, 256))

    # 计算指标
    metrics = compute_metrics(preds, targets, num_classes)

    print("Metrics:")
    for k, v in metrics.items():
        if not k.startswith('iou_class'):
            print(f"  {k}: {v:.4f}")

    # 测试指标跟踪器
    print("\nTesting MetricTracker...")
    tracker = MetricTracker(num_classes)

    for i in range(5):
        preds = torch.randint(0, num_classes, (2, 128, 128))
        targets = torch.randint(0, num_classes, (2, 128, 128))
        tracker.update(preds, targets)

    print(tracker.get_summary())
