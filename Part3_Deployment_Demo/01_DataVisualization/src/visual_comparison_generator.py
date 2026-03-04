"""
可视化对比图生成模块

生成 原始→增强→分割 的三联对比图。
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / '05_Shared'))

from models.real_models import mask_to_color_image


class VisualComparisonGenerator:
    """
    可视化对比图生成器

    生成用于展示和对比的可视化图像。
    """

    # SUIM类别颜色
    CLASS_COLORS = [
        (0, 0, 0),       # Background
        (255, 0, 0),     # Divers
        (0, 255, 0),     # Plants
        (0, 0, 255),     # Wrecks
        (255, 255, 0),   # Robots
        (255, 0, 255),   # Reefs
        (0, 255, 255),   # Fish
        (128, 128, 128)  # Sea floor
    ]

    CLASS_NAMES = [
        "Background",
        "Divers",
        "Plants",
        "Wrecks",
        "Robots",
        "Reefs",
        "Fish",
        "Sea_floor"
    ]

    def __init__(self, output_dir: str = None):
        """
        初始化生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir) if output_dir else Path('./output/comparisons')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_triple_comparison(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        mask: np.ndarray,
        labels: bool = True
    ) -> np.ndarray:
        """
        创建三联对比图 (原始 | 增强 | 分割)

        Args:
            original: 原始图像 [H, W, 3]
            enhanced: 增强后图像 [H, W, 3]
            mask: 分割mask [H, W]
            labels: 是否添加标签

        Returns:
            三联对比图 [H, 3*W, 3]
        """
        # 确保所有图像尺寸一致
        h, w = original.shape[:2]

        if enhanced.shape[:2] != (h, w):
            enhanced = cv2.resize(enhanced, (w, h))
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 转换mask为彩色
        color_mask = mask_to_color_image(mask, self.CLASS_COLORS)

        # 水平拼接
        comparison = np.hstack([original, enhanced, color_mask])

        # 添加标签
        if labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            color = (0, 255, 0)

            cv2.putText(comparison, "Original", (10, 30), font, font_scale, color, thickness)
            cv2.putText(comparison, "Enhanced", (w + 10, 30), font, font_scale, color, thickness)
            cv2.putText(comparison, "Segmented", (2 * w + 10, 30), font, font_scale, color, thickness)

        return comparison

    def create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        创建叠加图像

        Args:
            image: 原始/增强图像 [H, W, 3]
            mask: 分割mask [H, W]
            alpha: 叠加透明度

        Returns:
            叠加图像 [H, W, 3]
        """
        color_mask = mask_to_color_image(mask, self.CLASS_COLORS)

        # 调整尺寸
        if color_mask.shape[:2] != image.shape[:2]:
            color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]))

        overlay = (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)
        return overlay

    def create_grid_comparison(
        self,
        images: List[np.ndarray],
        masks: List[np.ndarray],
        grid_size: Tuple[int, int] = None
    ) -> np.ndarray:
        """
        创建网格对比图

        Args:
            images: 图像列表
            masks: mask列表
            grid_size: 网格大小 (rows, cols)

        Returns:
            网格对比图
        """
        n = len(images)
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
        else:
            rows, cols = grid_size

        # 获取单个图像尺寸
        h, w = images[0].shape[:2]
        cell_h, cell_w = h, w * 3  # 每个单元包含三联图

        # 创建空白画布
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

        # 填充网格
        for idx, (img, mask) in enumerate(zip(images, masks)):
            row = idx // cols
            col = idx % cols

            # 创建三联对比
            triple = self.create_triple_comparison(img, img, mask, labels=False)
            triple = cv2.resize(triple, (cell_w, cell_h))

            # 放入网格
            y_start = row * cell_h
            y_end = y_start + cell_h
            x_start = col * cell_w
            x_end = x_start + cell_w

            grid[y_start:y_end, x_start:x_end] = triple

        return grid

    def create_class_legend(
        self,
        size: Tuple[int, int] = (200, 300)
    ) -> np.ndarray:
        """
        创建类别图例

        Args:
            size: 图例尺寸 (width, height)

        Returns:
            图例图像
        """
        w, h = size
        legend = np.zeros((h, w, 3), dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        margin = 10
        box_size = 20

        y = margin + box_size
        for i, (color, name) in enumerate(zip(self.CLASS_COLORS, self.CLASS_NAMES)):
            # 绘制颜色框
            cv2.rectangle(legend, (margin, y - box_size),
                         (margin + box_size, y), color, -1)
            cv2.rectangle(legend, (margin, y - box_size),
                         (margin + box_size, y), (255, 255, 255), 1)

            # 绘制文字
            cv2.putText(legend, name, (margin + box_size + 5, y - 5),
                       font, font_scale, (255, 255, 255), thickness)

            y += box_size + 10

        return legend

    def save_comparison(
        self,
        comparison: np.ndarray,
        filename: str,
        add_legend: bool = True
    ) -> Path:
        """
        保存对比图

        Args:
            comparison: 对比图像
            filename: 文件名
            add_legend: 是否添加图例

        Returns:
            保存路径
        """
        # 添加图例
        if add_legend:
            legend = self.create_class_legend()
            legend_h, legend_w = legend.shape[:2]

            # 在右侧添加图例
            h, w = comparison.shape[:2]
            result = np.zeros((h, w + legend_w, 3), dtype=np.uint8)
            result[:, :w] = comparison
            result[:, w:] = legend

            comparison = result

        # 保存
        save_path = self.output_dir / filename
        cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        return save_path

    def create_statistics_text(
        self,
        mask: np.ndarray,
        original_size: Tuple[int, int] = None
    ) -> str:
        """
        创建分割统计文本

        Args:
            mask: 分割mask
            original_size: 原始图像尺寸 (H, W)

        Returns:
            统计文本
        """
        unique, counts = np.unique(mask, return_counts=True)
        total = mask.size

        lines = ["=== 分割统计 ==="]
        lines.append(f"图像尺寸: {mask.shape[1]} x {mask.shape[0]}")
        lines.append("检测到的类别:")

        for cls, cnt in zip(unique, counts):
            pct = cnt / total * 100
            cls_name = self.CLASS_NAMES[cls] if cls < len(self.CLASS_NAMES) else f"Class{cls}"
            lines.append(f"  {cls_name}: {cnt} pixels ({pct:.1f}%)")

        return "\n".join(lines)


if __name__ == '__main__':
    # 测试
    print("Testing VisualComparisonGenerator...")

    generator = VisualComparisonGenerator()

    # 创建测试图像
    original = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    enhanced = original.copy()
    mask = np.random.randint(0, 8, (256, 256), dtype=np.uint8)

    # 创建对比图
    comparison = generator.create_triple_comparison(original, enhanced, mask)
    print(f"Comparison shape: {comparison.shape}")

    # 创建图例
    legend = generator.create_class_legend()
    print(f"Legend shape: {legend.shape}")

    # 创建统计
    stats = generator.create_statistics_text(mask)
    print(stats)

    print("\n[OK] VisualComparisonGenerator test completed")
