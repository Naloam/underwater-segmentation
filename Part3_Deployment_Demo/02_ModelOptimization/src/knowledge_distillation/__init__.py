"""
知识蒸馏模块

实现从Mask2Former到轻量级模型的知识蒸馏。
"""

from .student_models import (
    LightweightSegmentor,
    DistillationLoss,
    create_student_model,
    STUDENT_MODEL_CONFIGS
)
from .distillation_trainer import (
    DistillationTrainer,
    extract_features_mask2former,
    extract_features_lightweight
)

__all__ = [
    'LightweightSegmentor',
    'DistillationLoss',
    'create_student_model',
    'STUDENT_MODEL_CONFIGS',
    'DistillationTrainer',
    'extract_features_mask2former',
    'extract_features_lightweight'
]
