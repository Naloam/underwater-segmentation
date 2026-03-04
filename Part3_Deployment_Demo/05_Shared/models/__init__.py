"""
共享模块 - 模型相关

包含Mock模型和统一模型接口，用于开发阶段的并行工作。
"""

from .mock_models import MockSegmentor, MockEnhancer
from .model_interface import SegmentationModelInterface, EnhancementModelInterface

__all__ = [
    'MockSegmentor',
    'MockEnhancer',
    'SegmentationModelInterface',
    'EnhancementModelInterface',
]
