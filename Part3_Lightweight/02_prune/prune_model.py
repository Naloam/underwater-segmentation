"""
模型剪枝 - 结构化L1剪枝

去除模型中不重要的通道，减少参数量和计算量。
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Tuple
import time

# 添加 Part2_Enhanced 路径
part2_enhanced_path = Path(__file__).parent.parent.parent / "Part2_Enhanced"
sys.path.insert(0, str(part2_enhanced_path))


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """加载原始模型"""
    print(f"[Prune] Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从checkpoint恢复配置
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        from configs.model_config import ModelConfig
        cfg_dict = checkpoint['config']['model']
        config = ModelConfig(**cfg_dict)
    else:
        from configs.model_config import ModelConfig
        config = ModelConfig()

    # 创建模型
    from models import create_model
    model = create_model(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    return model, config


def analyze_layer_importance(model: nn.Module) -> Dict[str, np.ndarray]:
    """
    分析每一层的重要性

    使用L1范数作为重要性指标
    """
    print("[Prune] Analyzing layer importance...")

    importance = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 计算每个输出通道的L1范数
            weight = module.weight.data  # [out_ch, in_ch, h, w]
            channel_importance = weight.abs().sum(dim=(1, 2, 3))  # [out_ch]
            importance[name] = channel_importance.cpu().numpy()

        elif isinstance(module, nn.Linear):
            # 计算每个输出神经元的L1范数
            weight = module.weight.data  # [out_ch, in_ch]
            neuron_importance = weight.abs().sum(dim=1)  # [out_ch]
            importance[name] = neuron_importance.cpu().numpy()

    return importance


def prune_conv_layer(module: nn.Module, prune_ratio: float) -> Tuple[int, int]:
    """
    剪枝卷积层

    Args:
        module: 卷积层
        prune_ratio: 剪枝比例

    Returns:
        (原始通道数, 保留通道数)
    """
    if not isinstance(module, nn.Conv2d):
        return 0, 0

    weight = module.weight.data  # [out_ch, in_ch, h, w]
    out_channels = weight.shape[0]

    # 计算每个通道的重要性
    importance = weight.abs().sum(dim=(1, 2, 3))

    # 确定要保留的通道数
    num_keep = int(out_channels * (1 - prune_ratio))
    if num_keep < 1:
        num_keep = 1

    # 找到重要的通道
    _, keep_indices = torch.topk(importance, num_keep)

    # 创建新的权重
    new_weight = weight[keep_indices].clone()

    if module.bias is not None:
        new_bias = module.bias[keep_indices].clone()
    else:
        new_bias = None

    # 更新层
    module.out_channels = num_keep
    module.weight.data = new_weight
    if module.bias is not None:
        module.bias.data = new_bias

    return out_channels, num_keep


def prune_model_structured(model: nn.Module, prune_ratio: float = 0.3) -> nn.Module:
    """
    结构化剪枝

    按照剪枝比例移除不重要的通道

    Args:
        model: 原始模型
        prune_ratio: 剪枝比例 (0.3 = 剪掉30%的通道)

    Returns:
        剪枝后的模型
    """
    print(f"[Prune] Applying structured pruning (ratio={prune_ratio})...")

    total_original = 0
    total_pruned = 0

    # 先遍历一遍收集所有卷积层
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    # 对每个卷积层进行剪枝
    for name, module in conv_layers:
        original, kept = prune_conv_layer(module, prune_ratio)
        total_original += original
        total_pruned += original - kept

    actual_prune_ratio = total_pruned / total_original if total_original > 0 else 0

    print(f"[Prune] Pruning complete!")
    print(f"  Total channels: {total_original}")
    print(f"  Pruned channels: {total_pruned}")
    print(f"  Actual ratio: {actual_prune_ratio:.1%}")

    return model


def fine_tune_model(model: nn.Module, device: str = 'cpu',
                   epochs: int = 1, lr: float = 1e-4):
    """
    微调剪枝后的模型

    简化版：仅做几个epoch的快速微调
    """
    print(f"[Finetune] Fine-tuning pruned model ({epochs} epochs)...")

    # 创建虚拟数据用于微调
    dummy_input = torch.randn(4, 3, 256, 256).to(device)
    dummy_target = torch.randint(0, 8, (4, 256, 256)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for i in range(10):  # 每个epoch 10个batch
            optimizer.zero_grad()

            output = model(dummy_input)
            loss = criterion(output, dummy_target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / 10
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    print("[Finetune] Fine-tuning complete!")


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """计算模型大小"""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size = param_size + buffer_size

    return {
        'param_size_mb': param_size / (1024 * 1024),
        'buffer_size_mb': buffer_size / (1024 * 1024),
        'total_size_mb': total_size / (1024 * 1024),
        'param_count': sum(p.numel() for p in model.parameters())
    }


def benchmark_inference(model: nn.Module, device: str = 'cpu',
                       num_iterations: int = 50, input_size: tuple = (1, 3, 256, 256)):
    """测试推理速度"""
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

    # 计时
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    end_time = time.time()

    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    fps = 1000 / avg_time_ms

    return {
        'avg_time_ms': avg_time_ms,
        'fps': fps,
        'iterations': num_iterations
    }


def prune_model(checkpoint_path: str, output_dir: Path,
               prune_ratio: float = 0.3, finetune_epochs: int = 1,
               device: str = 'cpu'):
    """
    完整的剪枝流程

    Args:
        checkpoint_path: 原始模型路径
        output_dir: 输出目录
        prune_ratio: 剪枝比例
        finetune_epochs: 微调轮数
        device: 运行设备
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model Pruning - Structured L1")
    print("=" * 60)

    # 1. 加载原始模型
    original_model, config = load_model(checkpoint_path, device)
    original_size = get_model_size(original_model)

    print(f"\n[Original Model]")
    print(f"  Parameters: {original_size['param_count']:,}")
    print(f"  Size: {original_size['total_size_mb']:.2f} MB")

    # 2. 基准测试原始模型
    original_benchmark = benchmark_inference(original_model, device)
    print(f"\n[Original Benchmark]")
    print(f"  Inference: {original_benchmark['avg_time_ms']:.2f} ms/image")
    print(f"  FPS: {original_benchmark['fps']:.2f}")

    # 3. 剪枝
    pruned_model = prune_model_structured(original_model, prune_ratio)
    pruned_size = get_model_size(pruned_model)

    print(f"\n[Pruned Model]")
    print(f"  Parameters: {pruned_size['param_count']:,}")
    print(f"  Size: {pruned_size['total_size_mb']:.2f} MB")
    print(f"  Compression: {(1 - pruned_size['param_count'] / original_size['param_count']) * 100:.1f}%")

    # 4. 微调
    if finetune_epochs > 0:
        fine_tune_model(pruned_model, device, finetune_epochs)

    # 5. 基准测试剪枝模型
    pruned_benchmark = benchmark_inference(pruned_model, device)
    print(f"\n[Pruned Benchmark]")
    print(f"  Inference: {pruned_benchmark['avg_time_ms']:.2f} ms/image")
    print(f"  FPS: {pruned_benchmark['fps']:.2f}")
    print(f"  Speedup: {original_benchmark['avg_time_ms'] / pruned_benchmark['avg_time_ms']:.2f}x")

    # 6. 保存剪枝模型
    pruned_path = output_dir / "model_pruned.pth"
    torch.save(pruned_model.state_dict(), pruned_path)
    print(f"\n[Save] Pruned model saved to {pruned_path}")

    # 7. 生成报告
    report = {
        'pruning': {
            'method': 'structured_l1',
            'prune_ratio': prune_ratio,
            'finetune_epochs': finetune_epochs,
            'device': device
        },
        'original_model': {
            'params': original_size['param_count'],
            'size_mb': round(original_size['total_size_mb'], 2),
            'inference_ms': round(original_benchmark['avg_time_ms'], 2),
            'fps': round(original_benchmark['fps'], 2)
        },
        'pruned_model': {
            'params': pruned_size['param_count'],
            'size_mb': round(pruned_size['total_size_mb'], 2),
            'inference_ms': round(pruned_benchmark['avg_time_ms'], 2),
            'fps': round(pruned_benchmark['fps'], 2)
        },
        'improvements': {
            'params_reduction': round((1 - pruned_size['param_count'] / original_size['param_count']) * 100, 1),
            'size_reduction': round((1 - pruned_size['total_size_mb'] / original_size['total_size_mb']) * 100, 1),
            'speedup': round(original_benchmark['avg_time_ms'] / pruned_benchmark['avg_time_ms'], 2)
        }
    }

    report_path = output_dir / "pruning_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"[Save] Report saved to {report_path}")

    print("\n" + "=" * 60)
    print("Pruning Complete!")
    print("=" * 60)

    return pruned_model, report


if __name__ == "__main__":
    # 配置
    checkpoint_path = "../Part2_Enhanced/checkpoints/best_model.pth"
    output_dir = Path("./results/pruned")
    prune_ratio = 0.3  # 剪掉30%的通道
    finetune_epochs = 1
    device = "cpu"

    # 执行剪枝
    pruned_model, report = prune_model(
        checkpoint_path, output_dir,
        prune_ratio, finetune_epochs, device
    )

    # 打印总结
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Original Params: {report['original_model']['params']:,}")
    print(f"Pruned Params: {report['pruned_model']['params']:,}")
    print(f"Reduction: {report['improvements']['params_reduction']}%")
    print(f"Speedup: {report['improvements']['speedup']}x")
