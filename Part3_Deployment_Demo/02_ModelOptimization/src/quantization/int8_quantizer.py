"""
INT8量化模块

实现PyTorch模型的INT8量化。
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from torch.utils.data import DataLoader


class INT8Quantizer:
    """
    INT8量化器

    支持动态量化和静态量化。
    """

    def __init__(
        self,
        model: nn.Module,
        backend: str = 'x86'
    ):
        """
        初始化量化器

        Args:
            model: 待量化的模型
            backend: 量化后端 ('x86', 'arm', 'fbgemm')
        """
        self.model = model
        self.backend = backend
        self.quantized_model = None

        # 设置量化后端
        try:
            torch.backends.quantized.engine = backend
        except:
            print(f"警告: 不支持后端 {backend}，使用默认后端")

    def dynamic_quantize(self) -> nn.Module:
        """
        动态量化

        适用于线性层和LSTM层，无需校准数据。

        Returns:
            量化后的模型
        """
        # 动态量化配置
        self.quantized_model = quant.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )

        return self.quantized_model

    def static_quantize(
        self,
        calibration_loader: DataLoader,
        num_calibration_batches: int = 10
    ) -> nn.Module:
        """
        静态量化

        需要校准数据集来确定激活值的量化参数。

        Args:
            calibration_loader: 校准数据加载器
            num_calibration_batches: 校准批次数

        Returns:
            量化后的模型
        """
        # 准备模型：融合操作
        self.model.eval()
        self.model.qconfig = quant.get_default_qconfig(self.backend)

        # 融合Conv+BN+ReLU等操作
        self.model = quant.fuse_modules(self.model)

        # 准备量化
        self.model = quant.prepare(self.model, inplace=True)

        # 校准
        print("开始校准...")
        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= num_calibration_batches:
                    break
                _ = self.model(batch['image'])

        # 转换为量化模型
        self.quantized_model = quant.convert(self.model, inplace=True)

        return self.quantized_model

    def qat_quantize(
        self,
        train_loader: DataLoader,
        num_epochs: int = 5,
        learning_rate: float = 1e-4
    ) -> nn.Module:
        """
        量化感知训练（QAT）

        在训练过程中模拟量化效果，获得更高的精度。

        Args:
            train_loader: 训练数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率

        Returns:
            量化后的模型
        """
        # 准备QAT
        self.model.train()
        self.model.qconfig = quant.get_default_qat_qconfig(self.backend)

        # 融合操作
        self.model = quant.fuse_modules(self.model)

        # 准备QAT
        self.model = quant.prepare_qat(self.model, inplace=True)

        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # 训练
        print("开始量化感知训练...")
        for epoch in range(num_epochs):
            for batch in train_loader:
                images = batch['image']
                labels = batch.get('label', batch.get('mask'))

                # 前向传播
                output = self.model(images)

                # 计算损失
                if labels is not None:
                    loss = nn.functional.cross_entropy(output, labels)
                else:
                    loss = output.mean()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"QAT Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        # 转换为量化模型
        self.model.eval()
        self.quantized_model = quant.convert(self.model, inplace=True)

        return self.quantized_model

    def get_quantization_stats(self) -> Dict[str, Any]:
        """获取量化统计信息"""
        if self.quantized_model is None:
            return {"error": "模型尚未量化"}

        # 计算模型大小
        def get_model_size(model):
            param_size = 0
            buffer_size = 0

            for param in model.parameters():
                param_size += param.numel() * param.element_size()

            for buffer in model.buffers():
                buffer_size += buffer.numel() * buffer.element_size()

            size_mb = (param_size + buffer_size) / (1024 ** 2)
            return size_mb

        original_size = get_model_size(self.model)
        quantized_size = get_model_size(self.quantized_model)

        # 计算压缩比
        compression_ratio = original_size / quantized_size

        # 统计量化层数
        quantized_layers = 0
        total_layers = 0

        for name, module in self.quantized_model.named_modules():
            if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Conv2d)):
                quantized_layers += 1
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                total_layers += 1

        return {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': compression_ratio,
            'size_reduction': (1 - 1/compression_ratio) * 100,
            'quantized_layers': quantized_layers,
            'total_layers': total_layers
        }

    def save_quantized_model(self, path: Path):
        """保存量化模型"""
        if self.quantized_model is None:
            raise ValueError("模型尚未量化")

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(torch.jit.script(self.quantized_model), str(path))
        print(f"量化模型已保存: {path}")


def quick_quantize(model: nn.Module, method: str = 'dynamic') -> nn.Module:
    """
    快速量化模型

    Args:
        model: 待量化的模型
        method: 量化方法 ('dynamic', 'static', 'qat')

    Returns:
        量化后的模型
    """
    quantizer = INT8Quantizer(model)

    if method == 'dynamic':
        return quantizer.dynamic_quantize()
    elif method == 'static':
        raise ValueError("静态量化需要校准数据")
    elif method == 'qat':
        raise ValueError("QAT需要训练数据")
    else:
        raise ValueError(f"未知的量化方法: {method}")


if __name__ == '__main__':
    print("INT8量化模块已就绪")
    print("等待真实模型进行测试...")
