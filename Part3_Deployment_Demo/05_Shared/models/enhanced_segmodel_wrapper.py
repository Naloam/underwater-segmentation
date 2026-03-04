"""
Part2 Enhanced 模型包装器

包装 Part2_Enhanced 训练好的增强型分割模型，实现统一接口。
"""

import sys
import os
from pathlib import Path

# 添加 Part2_Enhanced 路径 (必须在任何import之前)
part2_enhanced_path = Path(__file__).parent.parent.parent.parent / "Part2_Enhanced"
part2_enhanced_str = str(part2_enhanced_path)
if part2_enhanced_str not in sys.path:
    sys.path.insert(0, part2_enhanced_str)

# 临时切换到 Part2_Enhanced 目录以确保正确的模块导入
original_cwd = os.getcwd()
try:
    os.chdir(part2_enhanced_str)
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from typing import Dict, Any, Union
finally:
    os.chdir(original_cwd)

from .model_interface import SegmentationModelInterface


# 动态导入 Part2_Enhanced 模型的函数
def _load_part2_model(checkpoint_path, device):
    """从 Part2_Enhanced 加载模型"""
    import importlib.util

    # 加载 models/__init__.py
    models_spec = importlib.util.spec_from_file_location(
        "part2_models",
        part2_enhanced_path / "models" / "__init__.py"
    )
    part2_models = importlib.util.module_from_spec(models_spec)
    sys.modules['part2_models'] = part2_models
    models_spec.loader.exec_module(part2_models)

    # 加载 configs/model_config.py
    config_spec = importlib.util.spec_from_file_location(
        "part2_config",
        part2_enhanced_path / "configs" / "model_config.py"
    )
    part2_config = importlib.util.module_from_spec(config_spec)
    sys.modules['part2_config'] = part2_config
    config_spec.loader.exec_module(part2_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从checkpoint恢复配置
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        cfg_dict = checkpoint['config']['model']
        config = part2_config.ModelConfig(**cfg_dict)
    else:
        config = part2_config.ModelConfig()

    # 创建模型
    model = part2_models.create_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    return model, {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'miou': checkpoint.get('miou', 'N/A'),
    }


class EnhancedSegModelWrapper(SegmentationModelInterface):
    """
    Part2 Enhanced 分割模型包装器

    包装增强型模型 (CNN + CBAM + CLIP + Diffusion)，实现统一分割接口。
    """

    # SUIM数据集类别映射
    CLASS_NAMES = [
        "Background (waterbody)",
        "Human divers",
        "Plants and sea grass",
        "Wrecks and ruins",
        "Robots (AUVs/ROVs)",
        "Reefs and invertebrates",
        "Fish and vertebrates",
        "Sea floor and rocks"
    ]

    # 类别颜色映射（用于可视化）
    CLASS_COLORS = [
        (0, 0, 0),       # Background - 黑色
        (255, 0, 0),     # Divers - 红色
        (0, 255, 0),     # Plants - 绿色
        (0, 0, 255),     # Wrecks - 蓝色
        (255, 255, 0),   # Robots - 黄色
        (255, 0, 255),   # Reefs - 品红
        (0, 255, 255),   # Fish - 青色
        (128, 128, 128)  # Sea floor - 灰色
    ]

    def __init__(self, weight_path: str, num_classes: int = 8, device: str = 'cpu'):
        """
        初始化模型

        Args:
            weight_path: 模型权重路径 (Part2_Enhanced/checkpoints/best_model.pth)
            num_classes: 类别数
            device: 运行设备
        """
        self.device = device
        self.num_classes = num_classes
        self.input_size = (256, 256)

        # 动态导入 Part2_Enhanced 模型
        try:
            self.model, self.train_info = _load_part2_model(weight_path, device)
        except Exception as e:
            print(f"[!] 无法加载 Part2_Enhanced 模型: {e}")
            print(f"[!] 请确保 Part2_Enhanced 目录存在且包含所需模块")
            raise

        print(f"[OK] EnhancedSegModelWrapper initialized")
        print(f"   Device: {device}")
        print(f"   Weight: {weight_path}")
        print(f"   Trained: Epoch {self.train_info['epoch']}, mIoU {self.train_info['miou']}")

    def predict(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        预测单张图像的分割结果

        Args:
            image: 输入图像，可以是numpy数组或图像路径

        Returns:
            分割mask [H, W]，每个像素值为类别ID (0-7)
        """
        # 读取和预处理图像
        img_tensor, original_size = self._preprocess(image)

        # 推理
        with torch.no_grad():
            logits = self.model(img_tensor)  # [1, 8, 256, 256]

        # 后处理
        mask = logits.argmax(1).squeeze(0).cpu().numpy()  # [256, 256]

        # 调整回原始尺寸
        if original_size != (256, 256):
            mask = Image.fromarray(mask.astype(np.uint8))
            mask = mask.resize(original_size, Image.NEAREST)
            mask = np.array(mask)

        return mask

    def predict_batch(self, images: list) -> list:
        """
        批量预测图像的分割结果

        Args:
            images: 图像列表

        Returns:
            分割mask列表
        """
        results = []
        for img in images:
            results.append(self.predict(img))
        return results

    def _preprocess(self, image: Union[np.ndarray, str]) -> tuple:
        """
        图像预处理

        Args:
            image: 输入图像

        Returns:
            (tensor, original_size): 处理后的tensor和原始尺寸
        """
        # 读取图像
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
            original_size = img.size[::-1]  # (H, W)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
            original_size = image.shape[:2]  # (H, W)
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")

        # 调整尺寸到模型输入尺寸
        img_resized = img.resize((256, 256), Image.BILINEAR)

        # 转换为tensor [0, 1]
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        # 归一化 (ImageNet均值和标准差)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # 移到设备
        img_tensor = img_tensor.to(self.device)

        return img_tensor, original_size

    def get_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        params = sum(p.numel() for p in self.model.parameters())
        return {
            "name": "Enhanced SegModel (CNN+CBAM+CLIP+Diffusion)",
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "params": params,
            "params_M": params / 1e6,
            "device": self.device,
            "class_names": self.CLASS_NAMES,
            "train_epoch": self.train_info['epoch'],
            "train_miou": self.train_info['miou'],
        }

    def to(self, device: str):
        """将模型移动到指定设备"""
        self.device = device
        self.model = self.model.to(device)
        return self


# 辅助函数
def mask_to_color_image(mask: np.ndarray, colors: list = None) -> np.ndarray:
    """
    将分割mask转换为彩色图像

    Args:
        mask: 分割mask [H, W]
        colors: 类别颜色列表

    Returns:
        彩色图像 [H, W, 3]
    """
    if colors is None:
        colors = EnhancedSegModelWrapper.CLASS_COLORS

    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in enumerate(colors):
        color_img[mask == class_id] = color

    return color_img


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    将分割mask叠加到原图上

    Args:
        image: 原始图像 [H, W, 3]
        mask: 分割mask [H, W]
        alpha: 叠加透明度

    Returns:
        叠加后的图像 [H, W, 3]
    """
    color_mask = mask_to_color_image(mask)
    overlay = (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    return overlay


if __name__ == '__main__':
    # 测试代码
    print("测试 EnhancedSegModelWrapper...")

    # 创建模型
    wrapper = EnhancedSegModelWrapper(
        weight_path=str(Path(__file__).parent.parent.parent.parent / "Part2_Enhanced/checkpoints/best_model.pth"),
        num_classes=8,
        device='cpu'
    )

    # 打印模型信息
    info = wrapper.get_info()
    print(f"\n模型信息:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n EnhancedSegModelWrapper测试完成!")
