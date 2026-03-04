"""
Underwater Dataset

数据加载模块，支持SUIM、USIS10K、UIIS10K等数据集
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import numpy as np
from PIL import Image
import random


class UnderwaterDataset(Dataset):
    """
    水下图像数据集

    支持多种数据集格式，自动解析标注
    """
    def __init__(
        self,
        data_paths: List[str],
        image_size: Tuple[int, int] = (256, 256),
        split: str = 'train',
        transform: Optional[Callable] = None,
        num_classes: int = 8
    ):
        self.data_paths = data_paths
        self.image_size = image_size
        self.split = split
        self.transform = transform
        self.num_classes = num_classes

        # 收集所有图像
        self.samples = self._collect_samples()

        print(f"[Dataset] Loaded {len(self.samples)} samples for {split}")

    def _collect_samples(self) -> List[dict]:
        """收集所有图像和标注对"""
        samples = []

        for data_path in self.data_paths:
            data_root = Path(data_path)
            if not data_root.exists():
                print(f"[Warning] Path not found: {data_path}")
                continue

            # 查找图像文件
            image_dirs = []
            if 'raw' in [d.name for d in data_root.iterdir()]:
                # SUIM/USIS10K格式: 1_raw (图像), 6_label/masks (标签)
                raw_dir = data_root / '1_raw'
                mask_dir = data_root / '6_label' / 'masks'
                if raw_dir.exists() and mask_dir.exists():
                    image_dirs.append((raw_dir, mask_dir))
                    print(f"[Dataset] Found image/mask pair: {raw_dir} -> {mask_dir}")

            # 也支持直接读取所有图像
            for sub_dir in data_root.iterdir():
                if sub_dir.is_dir() and 'raw' in sub_dir.name.lower():
                    # 查找对应的mask目录
                    mask_candidates = list(data_root.glob('**/masks'))
                    if mask_candidates:
                        image_dirs.append((sub_dir, mask_candidates[0]))
                        print(f"[Dataset] Found image/mask pair: {sub_dir} -> {mask_candidates[0]}")

            # 收集样本
            for img_dir, mask_dir in image_dirs:
                img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

                for img_path in img_files:
                    # 查找对应的标注文件
                    label_name = img_path.stem
                    label_path = mask_dir / f'{label_name}.png'

                    if label_path.exists():
                        samples.append({
                            'image': str(img_path),
                            'label': str(label_path)
                        })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # 读取图像
        image = Image.open(sample['image']).convert('RGB')
        original_size = image.size

        # 读取标注
        label = Image.open(sample['label'])
        if label.mode != 'L':
            label = label.convert('L')

        # 调整尺寸
        image = image.resize(self.image_size, Image.BILINEAR)
        label = label.resize(self.image_size, Image.NEAREST)

        # 转换为tensor
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(np.array(label)).long()

        # 应用变换
        if self.transform:
            image, label = self.transform(image, label)

        return {
            'image': image,
            'label': label,
            'original_size': original_size,
            'path': sample['image']
        }


class SimpleTransform:
    """简单的数据变换"""
    def __init__(self, train: bool = True):
        self.train = train

    def __call__(self, image: torch.Tensor, label: torch.Tensor):
        if self.train:
            # 随机水平翻转
            if random.random() > 0.5:
                image = torch.flip(image, dims=[2])
                label = torch.flip(label, dims=[1])

            # 随机垂直翻转
            if random.random() > 0.5:
                image = torch.flip(image, dims=[1])
                label = torch.flip(label, dims=[0])

        return image, label


def create_dataloaders(
    train_paths: List[str],
    val_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    num_classes: int = 8
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器

    Args:
        train_paths: 训练数据路径列表
        val_path: 验证数据路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        image_size: 图像尺寸
        num_classes: 类别数

    Returns:
        train_loader, val_loader
    """
    # 训练数据集
    train_dataset = UnderwaterDataset(
        data_paths=train_paths,
        image_size=image_size,
        split='train',
        transform=SimpleTransform(train=True),
        num_classes=num_classes
    )

    # 验证数据集
    val_dataset = UnderwaterDataset(
        data_paths=[val_path],
        image_size=image_size,
        split='val',
        transform=SimpleTransform(train=False),
        num_classes=num_classes
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # 测试数据集
    print("Testing Underwater Dataset...")

    data_paths = [
        r"D:\myProjects\大创(1)\SUIM_Processed\SUIM_Processed",
        r"D:\myProjects\大创(1)\USIS10K_Processed\USIS10K_Processed"
    ]

    dataset = UnderwaterDataset(
        data_paths=data_paths,
        image_size=(256, 256),
        split='train'
    )

    print(f"Dataset size: {len(dataset)}")

    # 测试加载单个样本
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Label shape: {sample['label'].shape}")
    print(f"Label range: [{sample['label'].min()}, {sample['label'].max()}]")

    # 测试数据加载器
    train_loader, val_loader = create_dataloaders(
        train_paths=data_paths,
        val_path=data_paths[0],
        batch_size=2,
        num_workers=0
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
