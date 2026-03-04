"""
知识蒸馏训练器

实现从Mask2Former到轻量级模型的知识蒸馏。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import sys
from tqdm import tqdm

# 添加共享模块到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / '05_Shared'))

from .student_models import (
    LightweightSegmentor,
    DistillationLoss,
    create_student_model
)


class DistillationTrainer:
    """
    知识蒸馏训练器
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: str = 'cuda',
        temperature: float = 4.0,
        alpha: float = 0.5,
        lr: float = 1e-4
    ):
        """
        初始化训练器

        Args:
            teacher_model: 教师模型（已训练好的Mask2Former）
            student_model: 学生模型（轻量级模型）
            device: 训练设备
            temperature: 蒸馏温度
            alpha: 软标签损失权重
            lr: 学习率
        """
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.device = device

        # 教师模型设为评估模式
        self.teacher_model.eval()

        # 蒸馏损失
        self.criterion = DistillationLoss(
            alpha=alpha,
            temperature=temperature,
            feature_weight=1.0
        )

        # 优化器
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=lr,
            weight_decay=0.01
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6
        )

        # 训练统计
        self.epoch = 0
        self.best_loss = float('inf')

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        extract_features_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        单步训练

        Args:
            batch: 批次数据
            extract_features_fn: 特征提取函数（如果需要特征蒸馏）

        Returns:
            损失字典
        """
        images = batch['image'].to(self.device)
        labels = batch.get('label', batch.get('mask')).to(self.device)

        # 前向传播
        with torch.no_grad():
            teacher_output = self.teacher_model(images)
            teacher_features = None
            if extract_features_fn:
                teacher_features = extract_features_fn(self.teacher_model, images)

        student_output = self.student_model(images)
        student_features = None
        if extract_features_fn:
            student_features = extract_features_fn(self.student_model, images)

        # 计算损失
        loss, loss_details = self.criterion(
            student_output, teacher_output,
            student_features, teacher_features,
            labels
        )

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_details

    def train_epoch(
        self,
        train_loader: DataLoader,
        extract_features_fn: Optional[Callable] = None
    ) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            extract_features_fn: 特征提取函数

        Returns:
            平均损失
        """
        self.student_model.train()
        epoch_losses = {}

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            losses = self.train_step(batch, extract_features_fn)

            # 累积损失
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value)

            # 更新进度条
            pbar.set_postfix({k: sum(v)/len(v) for k, v in epoch_losses.items()})

        # 计算平均损失
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        return avg_losses

    def validate(self, val_loader: DataLoader) -> float:
        """
        验证

        Args:
            val_loader: 验证数据加载器

        Returns:
            验证损失
        """
        self.student_model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch.get('label', batch.get('mask')).to(self.device)

                # 教师模型输出
                teacher_output = self.teacher_model(images)

                # 学生模型输出
                student_output = self.student_model(images)

                # 计算损失
                loss, _ = self.criterion(student_output, teacher_output, None, None, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        checkpoint_dir: Optional[Path] = None,
        extract_features_fn: Optional[Callable] = None
    ) -> Dict[str, list]:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            checkpoint_dir: 检查点保存目录
            extract_features_fn: 特征提取函数

        Returns:
            训练历史
        """
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(num_epochs):
            self.epoch = epoch

            # 训练
            train_losses = self.train_epoch(train_loader, extract_features_fn)
            history['train_loss'].append(train_losses)

            # 验证
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)

                # 保存最佳模型
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    if checkpoint_dir:
                        self.save_checkpoint(checkpoint_dir / 'best_student.pth')

            # 学习率调度
            self.scheduler.step()

            # 保存检查点
            if checkpoint_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')

        return history

    def save_checkpoint(self, path: Path):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'student_model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"检查点已保存: {path}")

    def load_checkpoint(self, path: Path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.student_model.load_state_dict(checkpoint['student_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        print(f"检查点已加载: {path}")


# 模拟特征提取函数（实际使用时需要根据模型结构调整）
def extract_features_mask2former(model: nn.Module, x: torch.Tensor):
    """从Mask2Former提取特征（示例）"""
    features = []
    # 实际实现需要根据Mask2Former的结构
    # 这里返回空列表作为占位
    return features


def extract_features_lightweight(model: nn.Module, x: torch.Tensor):
    """从轻量级模型提取特征"""
    features = model.backbone(x)
    return features


if __name__ == '__main__':
    # 测试训练器
    print("知识蒸馏训练器模块已就绪")
    print("等待第二部分模型训练完成...")
