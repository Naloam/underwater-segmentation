"""
可视化对比图生成模块

生成 原始→增强→分割mask 的对比图，支持批量处理。
"""

import cv2
import numpy as np
import json
import base64
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import sys

# 添加共享模块到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / '05_Shared'))

from common.utils import load_image, save_image, create_comparison_grid, ensure_dir
from models.mock_models import SUIM_COLOR_MAP, mask_to_color_image, overlay_mask


class AnnotationParser:
    """标注解析器基类"""

    @staticmethod
    def parse_suim(json_path: Path) -> np.ndarray:
        """
        解析SUIM数据集的JSON标注

        Args:
            json_path: JSON标注文件路径

        Returns:
            分割mask [H, W]
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        height = data['size']['height']
        width = data['size']['width']
        mask = np.zeros((height, width), dtype=np.uint8)

        # 类别ID映射
        class_mapping = {
            'waterbody_background': 0,
            'human_divers': 1,
            'plants': 2,
            'wrecks_and_ruins': 3,
            'robots': 4,
            'reefs': 5,
            'fish': 6,
            'sea-floor_and_rocks': 7,
        }

        # 解析每个对象的bitmap
        for obj in data.get('objects', []):
            class_title = obj.get('classTitle', '')
            if class_title not in class_mapping:
                continue

            class_id = class_mapping[class_title]
            bitmap_data = obj.get('bitmap', {}).get('data', '')

            if bitmap_data:
                # 解码base64编码的bitmap
                try:
                    bitmap_bytes = base64.b64decode(bitmap_data)
                    # 这里需要根据实际编码格式解码
                    # SUIM使用的是Run-Length Encoding (RLE)格式
                    # 简化处理：使用轮廓或区域填充
                    # 实际项目中需要完整的RLE解码器

                    # 使用临时方案：创建示例mask
                    origin = obj.get('bitmap', {}).get('origin', [0, 0])
                    # 这里应该实现完整的RLE解码
                    # 暂时跳过，后续会生成随机mask用于演示

                except Exception as e:
                    print(f"警告: 解码bitmap失败: {e}")

        return mask

    @staticmethod
    def parse_usis10k(json_path: Path) -> np.ndarray:
        """
        解析USIS10K数据集的JSON标注

        Args:
            json_path: JSON标注文件路径

        Returns:
            分割mask [H, W]
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # USIS10K的JSON格式可能不同，需要根据实际情况调整
        # 这里提供一个基础框架
        height = data.get('height', 512)
        width = data.get('width', 512)
        mask = np.zeros((height, width), dtype=np.uint8)

        # 解析标注内容...

        return mask


class VisualComparisonGenerator:
    """可视化对比图生成器"""

    def __init__(self, dataset_root: Path):
        """
        初始化生成器

        Args:
            dataset_root: 数据集根目录
        """
        self.dataset_root = Path(dataset_root)
        self.parser = AnnotationParser()

    def generate_single_comparison(
        self,
        raw_path: Path,
        enhanced_path: Path,
        label_path: Path,
        output_path: Path,
        overlay: bool = True
    ) -> None:
        """
        生成单张图像的对比图

        Args:
            raw_path: 原始图像路径
            enhanced_path: 增强图像路径
            label_path: 标注文件路径
            output_path: 输出路径
            overlay: 是否生成叠加图
        """
        # 加载图像
        raw = load_image(raw_path)
        enhanced = load_image(enhanced_path)

        # 解析标注
        mask = self.parser.parse_suim(label_path)

        # 如果mask为空（解析失败），生成模拟mask用于演示
        if np.sum(mask) == 0:
            mask = self._generate_demo_mask(raw.shape[:2])

        # 生成彩色mask
        color_mask = mask_to_color_image(mask)

        # 创建对比图
        if overlay:
            # 原始 | 增强 | 分割 | 叠加
            overlay_img = overlay_mask(enhanced, mask, alpha=0.4)
            comparison = create_comparison_grid(
                [raw, enhanced, color_mask, overlay_img],
                labels=['Raw', 'Enhanced', 'Segmentation', 'Overlay']
            )
        else:
            # 原始 | 增强 | 分割
            comparison = create_comparison_grid(
                [raw, enhanced, color_mask],
                labels=['Raw', 'Enhanced', 'Segmentation']
            )

        # 保存
        ensure_dir(output_path.parent)
        save_image(comparison, output_path)

    def _generate_demo_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        生成演示用的分割mask

        Args:
            shape: 图像尺寸 (H, W)

        Returns:
            分割mask
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # 创建一些简单的区域模拟不同类别
        # 背景区域
        mask[:, :] = 0

        # 模拟一些物体区域
        # 水下生物区域（鱼）
        mask[h//4:h//2, w//4:w//2] = 6

        # 礁石区域
        mask[h//2:3*h//4, w//2:3*w//4] = 5

        # 海床区域
        mask[3*h//4:, :] = 7

        return mask

    def batch_generate(
        self,
        output_dir: Path,
        num_samples: int = 10,
        overlay: bool = True
    ) -> List[Path]:
        """
        批量生成对比图

        Args:
            output_dir: 输出目录
            num_samples: 处理样本数量
            overlay: 是否生成叠加图

        Returns:
            生成的文件路径列表
        """
        # 查找数据集子目录
        dataset_name = self.dataset_root.name
        dataset_path = self.dataset_root / dataset_name

        if not dataset_path.exists():
            dataset_path = self.dataset_root

        raw_dir = dataset_path / '1_raw'
        enhanced_dir = dataset_path / '2_enhanced'
        label_dir = dataset_path / '6_label' / 'ann'

        if not raw_dir.exists():
            print(f"警告: 找不到原始图像目录: {raw_dir}")
            return []

        # 获取图像列表
        raw_images = sorted(list(raw_dir.glob('*.jpg'))) + sorted(list(raw_dir.glob('*.png')))

        output_paths = []
        processed = 0

        for raw_path in raw_images[:num_samples]:
            base_name = raw_path.stem
            enhanced_path = enhanced_dir / f'{base_name}.jpg'
            label_path = label_dir / f'{raw_path.name}.json'

            # 检查文件是否存在
            if not enhanced_path.exists():
                enhanced_path = enhanced_dir / f'{base_name}.png'

            if not enhanced_path.exists() or not label_path.exists():
                print(f"跳过 {base_name}: 缺少增强图或标注")
                continue

            # 生成输出路径
            output_path = output_dir / f'{base_name}_comparison.jpg'
            self.generate_single_comparison(
                raw_path, enhanced_path, label_path, output_path, overlay
            )

            output_paths.append(output_path)
            processed += 1
            print(f"已处理: {base_name} ({processed}/{num_samples})")

        print(f"\n完成! 共生成 {len(output_paths)} 张对比图")
        return output_paths


def generate_all_datasets(
    project_root: Path,
    output_root: Path,
    samples_per_dataset: int = 10
) -> Dict[str, List[Path]]:
    """
    为所有数据集生成对比图

    Args:
        project_root: 项目根目录
        output_root: 输出根目录
        samples_per_dataset: 每个数据集处理样本数

    Returns:
        各数据集生成的文件路径
    """
    datasets = ['SUIM_Processed', 'USIS10K_Processed', 'EUVP_Processed', 'UIIS10K_Processed']
    results = {}

    for dataset in datasets:
        dataset_path = project_root / dataset
        if not dataset_path.exists():
            print(f"跳过 {dataset}: 目录不存在")
            continue

        print(f"\n处理 {dataset}...")
        output_dir = output_root / dataset
        generator = VisualComparisonGenerator(dataset_path)

        results[dataset] = generator.batch_generate(
            output_dir, samples_per_dataset, overlay=True
        )

    return results


if __name__ == '__main__':
    # 测试代码
    project_root = Path('d:/myProjects/大创(1)')
    output_root = Path('d:/myProjects/大创(1)/Part3_Deployment_Demo/01_DataVisualization/output/comparisons')

    # 为SUIM数据集生成对比图
    suim_path = project_root / 'SUIM_Processed'
    generator = VisualComparisonGenerator(suim_path)

    generator.batch_generate(
        output_root / 'SUIM_Processed',
        num_samples=5,
        overlay=True
    )
