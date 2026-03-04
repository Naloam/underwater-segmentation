"""
Mock模型 - 用于开发阶段

在第二部分模型训练完成前，使用Mock模型进行前端开发和测试。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
import cv2


class MockSegmentor(nn.Module):
    """
    模拟分割模型

    用于开发阶段的UI测试和功能验证。
    当真实模型训练完成后，可以通过统一接口无缝替换。
    """

    def __init__(self, num_classes: int = 8, input_size: Tuple[int, int] = (512, 512)):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = 'cpu'

        # 使用一个简单的卷积网络生成模拟结果
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            分割logits [B, num_classes, H, W]
        """
        B, _, H, W = x.shape
        output = self.conv(x)
        # 上采样到原始尺寸
        output = nn.functional.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)
        return output

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        预测单张图像

        Args:
            image: 输入图像 [H, W, 3] RGB格式

        Returns:
            分割mask [H, W]，每个像素值为类别ID
        """
        # 预处理
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)

        # 推理
        with torch.no_grad():
            output = self.forward(image_tensor)

        # 后处理
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        return pred

    def get_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": "MockSegmentor",
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "params": sum(p.numel() for p in self.parameters()),
            "device": self.device
        }


class MockEnhancer(nn.Module):
    """
    模拟图像增强模型

    用于开发阶段的UI测试。
    """

    def __init__(self):
        super().__init__()
        self.device = 'cpu'

        # 简单的增强网络：调整对比度和亮度
        self.contrast = nn.Parameter(torch.ones(1) * 1.2)
        self.brightness = nn.Parameter(torch.ones(1) * 10.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            增强后图像 [B, 3, H, W]
        """
        # 简单的对比度和亮度调整
        enhanced = x * self.contrast + self.brightness / 255.0
        return torch.clamp(enhanced, 0, 1)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        增强单张图像

        Args:
            image: 输入图像 [H, W, 3] RGB格式

        Returns:
            增强后图像 [H, W, 3]
        """
        # 预处理
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)

        # 推理
        with torch.no_grad():
            output = self.forward(image_tensor)

        # 后处理
        enhanced = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = (enhanced * 255).astype(np.uint8)
        return enhanced

    def get_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": "MockEnhancer",
            "contrast": self.contrast.item(),
            "brightness": self.brightness.item(),
            "params": sum(p.numel() for p in self.parameters()),
            "device": self.device
        }


class MockPipeline(nn.Module):
    """
    模拟完整流水线：增强 + 分割
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.enhancer = MockEnhancer()
        self.segmentor = MockSegmentor(num_classes=num_classes)
        self.device = 'cpu'

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            enhanced: 增强后图像 [B, 3, H, W]
            segmentation: 分割logits [B, num_classes, H, W]
        """
        enhanced = self.enhancer(x)
        segmentation = self.segmentor(enhanced)
        return enhanced, segmentation

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单张图像

        Args:
            image: 输入图像 [H, W, 3] RGB格式

        Returns:
            enhanced: 增强后图像 [H, W, 3]
            mask: 分割mask [H, W]
        """
        enhanced = self.enhancer.enhance(image)
        mask = self.segmentor.predict(enhanced)
        return enhanced, mask

    def get_info(self) -> Dict[str, Any]:
        """获取流水线信息"""
        return {
            "enhancer": self.enhancer.get_info(),
            "segmentor": self.segmentor.get_info(),
            "total_params": self.enhancer.get_info()["params"] + self.segmentor.get_info()["params"]
        }


# 颜色映射：用于可视化分割结果
SUIM_COLOR_MAP = {
    0: (0, 0, 0),        # 背景 - 黑色
    1: (128, 0, 0),      # human_divers - 深红
    2: (0, 128, 0),      # plants - 绿色
    3: (128, 128, 0),    # wrecks_and_ruins - 橄榄色
    4: (0, 0, 128),      # robots - 深蓝
    5: (128, 0, 128),    # reefs - 紫色
    6: (0, 128, 128),    # fish - 青色
    7: (128, 128, 128),  # sea-floor_and_rocks - 灰色
}


def mask_to_color_image(mask: np.ndarray, color_map: Dict[int, Tuple[int, int, int]] = None) -> np.ndarray:
    """
    将分割mask转换为彩色图像

    Args:
        mask: 分割mask [H, W]，每个像素值为类别ID
        color_map: 颜色映射表

    Returns:
        彩色图像 [H, W, 3] RGB格式
    """
    if color_map is None:
        color_map = SUIM_COLOR_MAP

    h, w = mask.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in color_map.items():
        color_image[mask == class_id] = color

    return color_image


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    将分割mask叠加到原图上

    Args:
        image: 原始图像 [H, W, 3] RGB格式
        mask: 分割mask [H, W]
        alpha: 叠加透明度

    Returns:
        叠加后的图像 [H, W, 3] RGB格式
    """
    color_mask = mask_to_color_image(mask)
    overlay = (image * (1 - alpha) + color_mask * alpha).astype(np.uint8)
    return overlay
