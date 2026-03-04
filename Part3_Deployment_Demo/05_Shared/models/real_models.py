"""
真实模型包装器

包装第二部分训练好的真实模型，实现统一的模型接口。
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Union
import sys

from .model_interface import (
    SegmentationModelInterface,
    EnhancementModelInterface,
    PipelineInterface
)
from .segmodel import SegModel, create_segmodel

# Windows编码兼容
def safe_print(msg):
    if sys.platform == 'win32':
        print(msg.encode('gbk', errors='ignore').decode('gbk'))
    else:
        print(msg)


class SegModelWrapper(SegmentationModelInterface):
    """
    Part2分割模型包装器

    包装SegModel，实现统一分割接口。
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

    def __init__(self, weight_path: str, num_classes: int = 8, device: str = 'cuda'):
        """
        初始化模型

        Args:
            weight_path: 模型权重路径
            num_classes: 类别数
            device: 运行设备
        """
        self.device = device
        self.num_classes = num_classes
        self.input_size = (256, 256)  # 模型固定的输入尺寸

        # 创建并加载模型
        self.model = create_segmodel(
            num_classes=num_classes,
            pretrained_path=weight_path,
            device=device
        )
        self.model.eval()

        safe_print("[OK] SegModelWrapper initialized")
        safe_print(f"   Device: {device}")
        safe_print(f"   Weight: {weight_path}")

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
            "name": "SegModel (Part2)",
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "params": params,
            "params_M": params / 1e6,
            "device": self.device,
            "class_names": self.CLASS_NAMES
        }

    def to(self, device: str):
        """将模型移动到指定设备"""
        self.device = device
        self.model = self.model.to(device)
        return self


class SimpleEnhancer(EnhancementModelInterface):
    """
    简单图像增强器

    目前使用基础增强方法，后续可替换为扩散模型。
    """

    def __init__(self, method='clahe'):
        """
        初始化增强器

        Args:
            method: 增强方法 ('clahe', 'he', 'none')
        """
        self.method = method
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            print("⚠️ OpenCV未安装，将使用基础增强")
            self.cv2 = None

    def enhance(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        增强单张图像

        Args:
            image: 输入图像

        Returns:
            增强后图像 [H, W, 3] RGB格式
        """
        # 读取图像
        if isinstance(image, str):
            img = np.array(Image.open(image))
        else:
            img = image.copy()

        # 转换为RGB
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # 应用增强
        if self.method == 'clahe' and self.cv2 is not None:
            # CLAHE增强
            lab = self.cv2.cvtColor(img, self.cv2.COLOR_RGB2LAB)
            l, a, b = self.cv2.split(lab)
            clahe = self.cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = self.cv2.merge([l, a, b])
            enhanced = self.cv2.cvtColor(enhanced, self.cv2.COLOR_LAB2RGB)
        elif self.method == 'he' and self.cv2 is not None:
            # 直方图均衡
            enhanced = self.cv2.cvtColor(img, self.cv2.COLOR_RGB2YCrCb)
            enhanced[:, :, 0] = self.cv2.equalizeHist(enhanced[:, :, 0])
            enhanced = self.cv2.cvtColor(enhanced, self.cv2.COLOR_YCrCb2RGB)
        else:
            # 无增强
            enhanced = img

        return enhanced

    def enhance_batch(self, images: list) -> list:
        """批量增强图像"""
        return [self.enhance(img) for img in images]

    def get_info(self) -> Dict[str, Any]:
        """获取增强器信息"""
        return {
            "name": "SimpleEnhancer",
            "method": self.method,
            "description": f"使用{self.method.upper()}方法进行图像增强"
        }


class SimplePipeline(PipelineInterface):
    """
    简单流水线（增强 + 分割）
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        初始化流水线

        Args:
            model_config: 模型配置
        """
        # 初始化增强器
        self.enhancer = SimpleEnhancer(method='clahe')

        # 初始化分割器
        weight_path = model_config.get('weight_path')
        device = model_config.get('device', 'cuda')
        self.segmentor = SegModelWrapper(
            weight_path=weight_path,
            num_classes=model_config.get('num_classes', 8),
            device=device
        )

    def process(self, image: Union[np.ndarray, str]) -> tuple:
        """
        处理单张图像

        Args:
            image: 输入图像

        Returns:
            (enhanced, mask): 增强后图像和分割mask
        """
        # 增强
        enhanced = self.enhancer.enhance(image)

        # 分割
        mask = self.segmentor.predict(enhanced)

        return enhanced, mask

    def get_info(self) -> Dict[str, Any]:
        """获取流水线信息"""
        return {
            "enhancer": self.enhancer.get_info(),
            "segmentor": self.segmentor.get_info()
        }


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
        colors = SegModelWrapper.CLASS_COLORS

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
    print("测试SegModelWrapper...")

    # 创建模型
    wrapper = SegModelWrapper(
        weight_path="checkpoints/trained/segmodel_best.pth",
        num_classes=8,
        device='cpu'
    )

    # 打印模型信息
    info = wrapper.get_info()
    print(f"\n模型信息:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n✅ SegModelWrapper测试完成!")
