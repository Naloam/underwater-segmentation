"""
量化模块

实现PyTorch模型的INT8量化。
"""

from .int8_quantizer import INT8Quantizer, quick_quantize

__all__ = ['INT8Quantizer', 'quick_quantize']
