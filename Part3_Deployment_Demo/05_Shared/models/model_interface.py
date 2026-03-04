"""
统一模型接口

定义统一的模型接口，确保Mock模型和真实模型可以无缝切换。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union
import numpy as np
import torch


class SegmentationModelInterface(ABC):
    """
    分割模型统一接口

    所有分割模型（真实模型或Mock模型）都需要实现此接口。
    """

    @abstractmethod
    def predict(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        预测单张图像的分割结果

        Args:
            image: 输入图像，可以是numpy数组或图像路径

        Returns:
            分割mask [H, W]，每个像素值为类别ID
        """
        pass

    @abstractmethod
    def predict_batch(self, images: list) -> list:
        """
        批量预测图像的分割结果

        Args:
            images: 图像列表

        Returns:
            分割mask列表
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            包含模型名称、参数量、设备等信息的字典
        """
        pass

    @abstractmethod
    def to(self, device: str):
        """
        将模型移动到指定设备

        Args:
            device: 设备名称，如 'cpu', 'cuda:0'
        """
        pass


class EnhancementModelInterface(ABC):
    """
    图像增强模型统一接口
    """

    @abstractmethod
    def enhance(self, image: Union[np.ndarray, str]) -> np.ndarray:
        """
        增强单张图像

        Args:
            image: 输入图像，可以是numpy数组或图像路径

        Returns:
            增强后图像 [H, W, 3] RGB格式
        """
        pass

    @abstractmethod
    def enhance_batch(self, images: list) -> list:
        """
        批量增强图像

        Args:
            images: 图像列表

        Returns:
            增强后图像列表
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            包含模型名称、参数量、设备等信息的字典
        """
        pass


class PipelineInterface(ABC):
    """
    完整流水线统一接口（增强 + 分割）
    """

    @abstractmethod
    def process(self, image: Union[np.ndarray, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单张图像

        Args:
            image: 输入图像

        Returns:
            enhanced: 增强后图像 [H, W, 3]
            mask: 分割mask [H, W]
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        获取流水线信息

        Returns:
            包含各模块信息的字典
        """
        pass


class ModelFactory:
    """
    模型工厂类

    根据配置创建对应的模型实例。
    """

    @staticmethod
    def create_segmentor(config: Dict[str, Any]) -> SegmentationModelInterface:
        """
        创建分割模型

        Args:
            config: 配置字典，包含：
                - use_mock: 是否使用Mock模型
                - weight_path: 真实模型权重路径（当use_mock=False时）
                - model_type: 模型类型 ('segmodel', 'enhanced', 或其他)

        Returns:
            分割模型实例
        """
        if config.get("use_mock", True):
            from .mock_models import MockSegmentor
            return MockSegmentor(
                num_classes=config.get("num_classes", 8),
                input_size=config.get("input_size", (512, 512))
            )
        else:
            model_type = config.get("model_type", "segmodel")

            if model_type == "enhanced":
                # 加载 Part2 Enhanced 模型
                from .enhanced_segmodel_wrapper import EnhancedSegModelWrapper
                return EnhancedSegModelWrapper(
                    weight_path=config.get("weight_path", "../Part2_Enhanced/checkpoints/best_model.pth"),
                    num_classes=config.get("num_classes", 8),
                    device=config.get("device", "cpu")
                )
            else:
                # 加载基础 SegModel (Part2)
                from .real_models import SegModelWrapper
                return SegModelWrapper(
                    weight_path=config.get("weight_path", "checkpoints/trained/segmodel_best.pth"),
                    num_classes=config.get("num_classes", 8),
                    device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
                )
#         if config.get("use_mock", True):
#     from .mock_models import MockSegmentor
#     return MockSegmentor(...)
# else:
#     from .real_models import Mask2FormerWrapper  # 新建这个文件
#     return Mask2FormerWrapper(
#         model_path=config.get("model_path"),
#         num_classes=config.get("num_classes", 8),
#         device=config.get("device", "cuda")
#     )


    @staticmethod
    def create_enhancer(config: Dict[str, Any]) -> EnhancementModelInterface:
        """
        创建增强模型

        Args:
            config: 配置字典

        Returns:
            增强模型实例
        """
        if config.get("use_mock", True):
            from .mock_models import MockEnhancer
            return MockEnhancer()
        else:
            # 加载真实增强器
            from .real_models import SimpleEnhancer
            return SimpleEnhancer(method=config.get("enhance_method", "clahe"))

    @staticmethod
    def create_pipeline(config: Dict[str, Any]) -> PipelineInterface:
        """
        创建完整流水线

        Args:
            config: 配置字典

        Returns:
            流水线实例
        """
        if config.get("use_mock", True):
            from .mock_models import MockPipeline
            return MockPipeline(num_classes=config.get("num_classes", 8))
        else:
            # 加载真实流水线
            from .real_models import SimplePipeline
            return SimplePipeline(model_config=config)
