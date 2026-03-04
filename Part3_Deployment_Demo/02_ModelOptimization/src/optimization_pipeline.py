"""
综合优化流水线

整合知识蒸馏、网络剪枝、INT8量化的完整优化流程。
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

# 添加共享模块到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / '05_Shared'))

from .knowledge_distillation import DistillationTrainer, create_student_model
from .pruning import auto_prune_model
from .quantization import INT8Quantizer


class OptimizationPipeline:
    """
    综合优化流水线

    执行顺序：知识蒸馏 → 网络剪枝 → INT8量化
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        device: str = 'cuda',
        output_dir: Path = None
    ):
        """
        初始化优化流水线

        Args:
            teacher_model: 教师模型（训练好的Mask2Former）
            device: 训练设备
            output_dir: 输出目录
        """
        self.teacher_model = teacher_model
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path('./optimization_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 记录优化历史
        self.history = {
            'original_model': {},
            'distillation': {},
            'pruning': {},
            'quantization': {},
            'final': {}
        }

    def run_full_pipeline(
        self,
        train_loader,
        val_loader,
        num_classes: int = 8,
        distill_epochs: int = 50,
        prune_ratio: float = 0.3,
        quantization_method: str = 'dynamic'
    ) -> Dict[str, Any]:
        """
        运行完整的优化流水线

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_classes: 类别数
            distill_epochs: 蒸馏训练轮数
            prune_ratio: 剪枝比例
            quantization_method: 量化方法

        Returns:
            优化结果和统计信息
        """
        print("="*60)
        print("开始综合优化流水线")
        print("="*60)

        # 步骤1: 知识蒸馏
        print("\n[步骤1/3] 知识蒸馏...")
        student_model = create_student_model('mobilenet_base')

        trainer = DistillationTrainer(
            self.teacher_model,
            student_model,
            device=self.device
        )

        history = trainer.fit(
            train_loader,
            val_loader,
            num_epochs=distill_epochs,
            checkpoint_dir=self.output_dir / 'checkpoints'
        )

        distilled_model = trainer.student_model

        self.history['distillation'] = {
            'final_loss': history['train_loss'][-1],
            'best_val_loss': trainer.best_loss
        }

        # 步骤2: 网络剪枝
        print("\n[步骤2/3] 网络剪枝...")
        pruned_model, prune_stats = auto_prune_model(
            distilled_model,
            data_loader=val_loader,
            target_prune_ratio=prune_ratio,
            method='bn_scale'
        )

        self.history['pruning'] = prune_stats

        # 微调剪枝后的模型
        print("微调剪枝后的模型...")
        fine_tuner = DistillationTrainer(
            self.teacher_model,
            pruned_model,
            device=self.device,
            lr=1e-5
        )

        fine_tuner.fit(
            train_loader,
            val_loader,
            num_epochs=10
        )

        pruned_model = fine_tuner.student_model

        # 步骤3: INT8量化
        print("\n[步骤3/3] INT8量化...")
        quantizer = INT8Quantizer(pruned_model, backend='x86')

        if quantization_method == 'dynamic':
            quantized_model = quantizer.dynamic_quantize()
        elif quantization_method == 'static':
            quantized_model = quantizer.static_quantize(train_loader)
        elif quantization_method == 'qat':
            quantized_model = quantizer.qat_quantize(train_loader, num_epochs=5)

        quant_stats = quantizer.get_quantization_stats()
        self.history['quantization'] = quant_stats

        # 保存最终模型
        print("\n保存最终模型...")
        final_model_path = self.output_dir / 'final_optimized_model.pth'
        torch.save(quantized_model.state_dict(), final_model_path)

        # 量化模型也保存为TorchScript
        quantizer.save_quantized_model(
            self.output_dir / 'final_optimized_model_jit.pth'
        )

        # 生成优化报告
        self._generate_report()

        print("\n"+"="*60)
        print("优化流水线完成!")
        print("="*60)

        return self.history

    def _generate_report(self):
        """生成优化报告"""
        report_path = self.output_dir / 'optimization_report.json'

        report = {
            'timestamp': datetime.now().isoformat(),
            'history': self.history,
            'summary': {
                'pipeline': 'distillation -> pruning -> quantization',
                'target_model': 'LightweightSegmentor',
                'final_size_mb': self.history['quantization'].get('quantized_size_mb', 0),
                'compression_ratio': self.history['quantization'].get('compression_ratio', 0),
                'params_pruned': self.history['pruning'].get('pruned_params', 0),
            }
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"优化报告已保存: {report_path}")

        # 打印摘要
        print("\n--- 优化摘要 ---")
        print(f"原始参数量: {self.history['pruning'].get('total_params', 0) / 1e6:.2f} M")
        print(f"剪枝比例: {self.history['pruning'].get('pruning_ratio', 0) * 100:.1f}%")
        print(f"最终模型大小: {self.history['quantization'].get('quantized_size_mb', 0):.2f} MB")
        print(f"压缩比: {self.history['quantization'].get('compression_ratio', 0):.2f}x")


# 预定义的优化配置
OPTIMIZATION_CONFIGS = {
    'aggressive': {
        'prune_ratio': 0.5,
        'quantization': 'dynamic',
        'distill_epochs': 30
    },
    'balanced': {
        'prune_ratio': 0.3,
        'quantization': 'dynamic',
        'distill_epochs': 50
    },
    'conservative': {
        'prune_ratio': 0.2,
        'quantization': 'static',
        'distill_epochs': 50
    }
}


if __name__ == '__main__':
    print("综合优化流水线模块已就绪")
    print("等待第二部分模型训练完成...")
