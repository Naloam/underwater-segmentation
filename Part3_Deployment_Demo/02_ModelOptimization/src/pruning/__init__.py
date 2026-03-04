"""
网络剪枝模块

实现基于重要性分析的结构化剪枝。
"""

from .channel_pruner import (
    ChannelImportanceAnalyzer,
    ChannelPruner,
    auto_prune_model
)

__all__ = [
    'ChannelImportanceAnalyzer',
    'ChannelPruner',
    'auto_prune_model'
]
