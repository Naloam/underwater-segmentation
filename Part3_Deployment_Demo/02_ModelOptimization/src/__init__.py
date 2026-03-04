"""
模型优化模块

包含知识蒸馏、网络剪枝、INT8量化和综合优化流水线。
"""

from .knowledge_distillation import (
    LightweightSegmentor,
    DistillationTrainer,
    create_student_model
)
from .pruning import auto_prune_model
from .quantization import INT8Quantizer
from .optimization_pipeline import OptimizationPipeline, OPTIMIZATION_CONFIGS

__all__ = [
    'LightweightSegmentor',
    'DistillationTrainer',
    'create_student_model',
    'auto_prune_model',
    'INT8Quantizer',
    'OptimizationPipeline',
    'OPTIMIZATION_CONFIGS'
]
