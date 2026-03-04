"""
通道剪枝模块

实现基于重要性分析的结构化剪枝。
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


class ChannelImportanceAnalyzer:
    """
    通道重要性分析器

    使用多种方法分析卷积层中各通道的重要性。
    """

    def __init__(self, method: str = 'bn_scale'):
        """
        初始化分析器

        Args:
            method: 分析方法
                - 'bn_scale': 基于BatchNorm的缩放因子
                - 'l1_norm': 基于L1范数
                - 'taylor': 基于Taylor展开
        """
        self.method = method
        self.importance_scores = {}

    def analyze_model(
        self,
        model: nn.Module,
        data_loader = None,
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        分析整个模型的通道重要性

        Args:
            model: 待分析的模型
            data_loader: 数据加载器（某些方法需要）
            num_samples: 采样数量

        Returns:
            重要性分数字典 {layer_name: scores}
        """
        model.eval()

        if self.method == 'bn_scale':
            self._analyze_by_bn_scale(model)
        elif self.method == 'l1_norm':
            self._analyze_by_l1_norm(model, data_loader, num_samples)
        elif self.method == 'taylor':
            self._analyze_by_taylor(model, data_loader, num_samples)

        return self.importance_scores

    def _analyze_by_bn_scale(self, model: nn.Module):
        """基于BatchNorm缩放因子分析"""
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # 使用缩放因子作为重要性
                scores = torch.abs(module.weight.data)
                self.importance_scores[name] = scores

    def _analyze_by_l1_norm(
        self,
        model: nn.Module,
        data_loader,
        num_samples: int
    ):
        """基于L1范数分析"""
        # 收集激活值
        activation_stats = {}

        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    avg_activation = torch.mean(torch.abs(output), dim=[0, 2, 3])
                    if name not in activation_stats:
                        activation_stats[name] = []
                    activation_stats[name].append(avg_activation)
            return hook

        # 注册hook
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)

        # 收集数据
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_samples:
                    break
                _ = model(batch['image'])

        # 计算平均激活
        for name, activations in activation_stats.items():
            scores = torch.stack(activations).mean(dim=0)
            self.importance_scores[name] = scores

        # 移除hook
        for hook in hooks:
            hook.remove()

    def _analyze_by_taylor(
        self,
        model: nn.Module,
        data_loader,
        num_samples: int
    ):
        """基于Taylor展开分析（考虑梯度）"""
        model.train()

        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break

            images = batch['image']
            labels = batch.get('label', batch.get('mask'))

            # 前向传播
            output = model(images)

            # 计算损失
            if labels is not None:
                loss = nn.functional.cross_entropy(output, labels)
            else:
                loss = output.mean()

            # 反向传播
            loss.backward()

            # 计算重要性：|梯度 * 激活|
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    if module.weight.grad is not None:
                        gradient = torch.abs(module.weight.grad)
                        weight = torch.abs(module.weight.data)
                        scores = torch.mean(gradient * weight, dim=[1, 2, 3])

                        if name not in self.importance_scores:
                            self.importance_scores[name] = []
                        self.importance_scores[name].append(scores)

            # 清零梯度
            model.zero_grad()

        # 平均重要性
        for name, scores_list in self.importance_scores.items():
            self.importance_scores[name] = torch.stack(scores_list).mean(dim=0)


class ChannelPruner:
    """
    通道剪枝器

    执行结构化剪枝，移除不重要的通道。
    """

    def __init__(
        self,
        model: nn.Module,
        importance_scores: Dict[str, torch.Tensor],
        prune_ratio: float = 0.3
    ):
        """
        初始化剪枝器

        Args:
            model: 待剪枝的模型
            importance_scores: 重要性分数
            prune_ratio: 剪枝比例
        """
        self.model = model
        self.importance_scores = importance_scores
        self.prune_ratio = prune_ratio

    def prune_channels(self) -> nn.Module:
        """
        执行通道剪枝

        Returns:
            剪枝后的模型
        """
        # 为每个卷积层生成剪枝mask
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 获取重要性分数
                scores = self.importance_scores.get(name, None)

                if scores is not None and len(scores) == module.out_channels:
                    # 计算剪枝阈值
                    num_prune = int(module.out_channels * self.prune_ratio)
                    threshold = torch.topk(scores, num_prune, largest=False).values[-1]

                    # 生成mask
                    mask = scores > threshold

                    # 应用剪枝
                    prune.custom_from_mask(
                        module,
                        name='weight',
                        mask=mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    )

        return self.model

    def get_pruning_stats(self) -> Dict[str, Any]:
        """获取剪枝统计信息"""
        total_params = 0
        remaining_params = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                total_params += module.weight.numel()
                remaining_params += module.weight.count_nonzero().item()

        pruning_ratio = 1 - (remaining_params / total_params)

        return {
            'total_params': total_params,
            'remaining_params': remaining_params,
            'pruned_params': total_params - remaining_params,
            'pruning_ratio': pruning_ratio
        }


def auto_prune_model(
    model: nn.Module,
    data_loader = None,
    target_prune_ratio: float = 0.3,
    method: str = 'bn_scale'
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    自动剪枝模型

    Args:
        model: 待剪枝的模型
        data_loader: 数据加载器
        target_prune_ratio: 目标剪枝比例
        method: 重要性分析方法

    Returns:
        剪枝后的模型, 统计信息
    """
    # 分析重要性
    analyzer = ChannelImportanceAnalyzer(method=method)
    importance_scores = analyzer.analyze_model(model, data_loader)

    # 执行剪枝
    pruner = ChannelPruner(model, importance_scores, target_prune_ratio)
    pruned_model = pruner.prune_channels()

    # 获取统计信息
    stats = pruner.get_pruning_stats()

    return pruned_model, stats


if __name__ == '__main__':
    print("通道剪枝模块已就绪")
    print("等待真实模型进行测试...")
