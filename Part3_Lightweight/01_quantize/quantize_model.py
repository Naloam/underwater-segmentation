"""
模型量化 - INT8动态量化

将训练好的FP32模型量化为INT8，减少模型体积并加速推理。
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, Any
import time

# 添加 Part2_Enhanced 路径
part2_enhanced_path = Path(__file__).parent.parent.parent / "Part2_Enhanced"
sys.path.insert(0, str(part2_enhanced_path))


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """加载原始模型"""
    print(f"[Quantize] Loading model from {checkpoint_path}")

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


def quantize_dynamic(model: nn.Module, device: str = 'cpu') -> nn.Module:
    """
    动态量化

    将线性层、卷积层等量化为INT8
    """
    print("[Quantize] Applying dynamic INT8 quantization...")

    # 动态量化：只量化权重，推理时动态量化激活值
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d, nn.BatchNorm2d},  # 量化的层类型
        dtype=torch.qint8  # INT8量化
    )

    print("[Quantize] Quantization complete!")
    return quantized_model


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """计算模型大小"""
    # 计算参数量
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
    print(f"[Benchmark] Testing inference speed ({num_iterations} iterations)...")

    model.eval()
    dummy_input = torch.randn(*input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 计时
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / num_iterations * 1000
    fps = 1000 / avg_time_ms

    return {
        'avg_time_ms': avg_time_ms,
        'fps': fps,
        'iterations': num_iterations
    }


def compare_outputs(original_model: nn.Module, quantized_model: nn.Module,
                   device: str = 'cpu', num_samples: int = 10):
    """比较原始模型和量化模型的输出差异"""
    print("[Compare] Comparing outputs between original and quantized models...")

    original_model.eval()
    quantized_model.eval()

    max_diffs = []
    mean_diffs = []

    for i in range(num_samples):
        dummy_input = torch.randn(1, 3, 256, 256).to(device)

        with torch.no_grad():
            original_output = original_model(dummy_input)
            quantized_output = quantized_model(dummy_input)

        # 计算差异
        diff = (original_output - quantized_output).abs()
        max_diffs.append(diff.max().item())
        mean_diffs.append(diff.mean().item())

    return {
        'max_diff': np.mean(max_diffs),
        'mean_diff': np.mean(mean_diffs),
        'samples': num_samples
    }


def quantize_model(checkpoint_path: str, output_dir: Path, device: str = 'cpu'):
    """
    完整的量化流程

    Args:
        checkpoint_path: 原始模型路径
        output_dir: 输出目录
        device: 运行设备
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model Quantization - INT8 Dynamic")
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

    # 3. 量化
    quantized_model = quantize_dynamic(original_model, device)
    quantized_size = get_model_size(quantized_model)

    print(f"\n[Quantized Model]")
    print(f"  Parameters: {quantized_size['param_count']:,}")
    print(f"  Size: {quantized_size['total_size_mb']:.2f} MB")
    print(f"  Compression: {(1 - quantized_size['total_size_mb'] / original_size['total_size_mb']) * 100:.1f}%")

    # 4. 基准测试量化模型
    quantized_benchmark = benchmark_inference(quantized_model, device)
    print(f"\n[Quantized Benchmark]")
    print(f"  Inference: {quantized_benchmark['avg_time_ms']:.2f} ms/image")
    print(f"  FPS: {quantized_benchmark['fps']:.2f}")
    print(f"  Speedup: {original_benchmark['avg_time_ms'] / quantized_benchmark['avg_time_ms']:.2f}x")

    # 5. 比较输出
    output_diff = compare_outputs(original_model, quantized_model, device)
    print(f"\n[Output Comparison]")
    print(f"  Max diff: {output_diff['max_diff']:.6f}")
    print(f"  Mean diff: {output_diff['mean_diff']:.6f}")

    # 6. 保存量化模型
    quantized_path = output_dir / "model_quantized.pth"
    torch.save(quantized_model.state_dict(), quantized_path)
    print(f"\n[Save] Quantized model saved to {quantized_path}")

    # 7. 生成报告
    report = {
        'quantization': {
            'method': 'dynamic_int8',
            'device': device
        },
        'original_model': {
            'size_mb': round(original_size['total_size_mb'], 2),
            'params': original_size['param_count'],
            'inference_ms': round(original_benchmark['avg_time_ms'], 2),
            'fps': round(original_benchmark['fps'], 2)
        },
        'quantized_model': {
            'size_mb': round(quantized_size['total_size_mb'], 2),
            'params': quantized_size['param_count'],
            'inference_ms': round(quantized_benchmark['avg_time_ms'], 2),
            'fps': round(quantized_benchmark['fps'], 2)
        },
        'improvements': {
            'compression_percent': round((1 - quantized_size['total_size_mb'] / original_size['total_size_mb']) * 100, 1),
            'speedup': round(original_benchmark['avg_time_ms'] / quantized_benchmark['avg_time_ms'], 2)
        },
        'accuracy': {
            'max_diff': round(output_diff['max_diff'], 6),
            'mean_diff': round(output_diff['mean_diff'], 6)
        }
    }

    report_path = output_dir / "quantization_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"[Save] Report saved to {report_path}")

    print("\n" + "=" * 60)
    print("Quantization Complete!")
    print("=" * 60)

    return quantized_model, report


if __name__ == "__main__":
    # 配置
    checkpoint_path = "../Part2_Enhanced/checkpoints/best_model.pth"
    output_dir = Path("./results/quantized")
    device = "cpu"

    # 执行量化
    quantized_model, report = quantize_model(checkpoint_path, output_dir, device)

    # 打印总结
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Original Size: {report['original_model']['size_mb']} MB")
    print(f"Quantized Size: {report['quantized_model']['size_mb']} MB")
    print(f"Compression: {report['improvements']['compression_percent']}%")
    print(f"Speedup: {report['improvements']['speedup']}x")
    print(f"Accuracy Loss: {report['accuracy']['mean_diff']:.6f}")
