"""
Visualization Module

可视化预测结果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Tuple, Optional


# 类别颜色（SUIM数据集）
CLASS_COLORS = [
    [0, 0, 0],        # Background (waterbody) - Black
    [255, 0, 0],      # Human divers - Red
    [0, 255, 0],      # Plants and sea grass - Green
    [0, 0, 255],      # Wrecks and ruins - Blue
    [255, 255, 0],    # Robots (AUVs/ROVs) - Yellow
    [255, 0, 255],    # Reefs and invertebrates - Magenta
    [0, 255, 255],    # Fish and vertebrates - Cyan
    [128, 128, 128],  # Sea floor and rocks - Gray
]

CLASS_NAMES = [
    "Background",
    "Divers",
    "Plants",
    "Wrecks",
    "Robots",
    "Reefs",
    "Fish",
    "Sea floor",
]


def label_to_color(label: torch.Tensor or np.ndarray) -> np.ndarray:
    """
    将标签转换为彩色图像

    Args:
        label: [H, W] 标签
    Returns:
        color: [H, W, 3] RGB彩色图像
    """
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    h, w = label.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color_rgb in enumerate(CLASS_COLORS):
        mask = label == cls
        color[mask] = color_rgb

    return color


def overlay_mask(
    image: torch.Tensor or np.ndarray,
    mask: torch.Tensor or np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    将掩码叠加到图像上

    Args:
        image: [H, W, 3] RGB图像
        mask: [H, W] 标签
        alpha: 叠加透明度
    Returns:
        overlay: [H, W, 3] 叠加后的图像
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    # 转换图像格式
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # 生成彩色掩码
    color_mask = label_to_color(mask)

    # 叠加
    overlay = (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    return overlay


def visualize_prediction(
    image: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    class_names: List[str] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    可视化预测结果

    Args:
        image: [C, H, W] 输入图像
        target: [H, W] 目标标签
        pred: [H, W] 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
    Returns:
        fig: matplotlib图表
    """
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:  # [B, C, H, W]
            image = image[0]  # 取第一张图
        if image.dim() == 3:  # [C, H, W]
            image = image.cpu().numpy().transpose(1, 2, 0)
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
        if target.ndim == 3:  # [B, H, W]
            target = target[0]  # 取第一张
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
        if pred.ndim == 3:  # [B, H, W]
            pred = pred[0]  # 取第一张

    if class_names is None:
        class_names = CLASS_NAMES

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 原始图像
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 目标标签
    target_color = label_to_color(target)
    axes[0, 1].imshow(target_color)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')

    # 预测标签
    pred_color = label_to_color(pred)
    axes[1, 0].imshow(pred_color)
    axes[1, 0].set_title('Prediction')
    axes[1, 0].axis('off')

    # 叠加
    overlay = overlay_mask(image, pred)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def save_comparison(
    images: List[torch.Tensor],
    targets: List[torch.Tensor],
    preds: List[torch.Tensor],
    output_dir: Path,
    prefix: str = "comparison"
):
    """
    批量保存对比图

    Args:
        images: 图像列表
        targets: 目标标签列表
        preds: 预测标签列表
        output_dir: 输出目录
        prefix: 文件前缀
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, tgt, pred) in enumerate(zip(images, targets, preds)):
        save_path = output_dir / f"{prefix}_{i:04d}.png"
        fig = visualize_prediction(img, tgt, pred, save_path=save_path)
        plt.close(fig)

    print(f"[Vis] Saved {len(images)} comparisons to {output_dir}")


def create_legend() -> plt.Figure:
    """创建类别颜色图例"""
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        color_norm = [c / 255 for c in color]
        ax.barh(i, 1, color=color_norm, label=name)
        ax.text(0.5, i, name, va='center', ha='left',
                color='white' if sum(color) < 400 else 'black',
                fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(CLASS_NAMES) - 0.5)
    ax.axis('off')
    ax.set_title('Class Legend', fontsize=14, fontweight='bold')

    return fig


def plot_metrics(
    metrics_history: dict,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    绘制训练曲线

    Args:
        metrics_history: 指标历史字典
        save_path: 保存路径
    Returns:
        fig: matplotlib图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 损失曲线
    if 'train_loss' in metrics_history:
        axes[0, 0].plot(metrics_history['train_loss'], label='Train')
    if 'val_loss' in metrics_history:
        axes[0, 0].plot(metrics_history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # mIoU曲线
    if 'miou' in metrics_history:
        axes[0, 1].plot(metrics_history['miou'], label='mIoU', color='green')
    axes[0, 1].set_title('Mean IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].grid(True, alpha=0.3)

    # 准确率曲线
    if 'accuracy' in metrics_history:
        axes[1, 0].plot(metrics_history['accuracy'], label='Accuracy', color='blue')
    axes[1, 0].set_title('Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].grid(True, alpha=0.3)

    # 学习率曲线
    if 'lr' in metrics_history:
        axes[1, 1].plot(metrics_history['lr'], label='LR', color='red')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # 测试可视化
    print("Testing Visualization...")

    # 模拟数据
    image = torch.rand(3, 256, 256)
    target = torch.randint(0, 8, (256, 256))
    pred = torch.randint(0, 8, (256, 256))

    # 可视化
    fig = visualize_prediction(image, target, pred)
    plt.show()

    # 测试批量保存
    output_dir = Path("./test_vis")
    images = [image] * 3
    targets = [target] * 3
    preds = [pred] * 3

    save_comparison(images, targets, preds, output_dir)

    # 测试图例
    legend_fig = create_legend()
    plt.show()
