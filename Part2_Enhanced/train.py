"""
Training Script

Part2 Enhanced - 训练脚本
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from configs.model_config import ModelConfig, TrainingConfig, get_config
from models import create_model
from data import create_dataloaders
from losses import CombinedLoss


class Trainer:
    """训练器"""
    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainingConfig
    ):
        self.model_config = model_config
        self.train_config = train_config

        # 设备
        self.device = torch.device(train_config.device)

        # 创建模型
        self.model = create_model(model_config).to(self.device)
        print(f"[Model] Created with {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M parameters")

        # 损失函数
        self.criterion = CombinedLoss(
            num_classes=model_config.num_classes,
            ce_weight=model_config.ce_loss_weight,
            focal_weight=0.5,
            dice_weight=1.0,
            pq_weight=model_config.pq_loss_weight,
            semantic_weight=model_config.semantic_loss_weight
        )

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_config.epochs,
            eta_min=1e-6
        )

        # TensorBoard (可选)
        if HAS_TENSORBOARD:
            log_dir = Path(train_config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        # 训练状态
        self.current_epoch = 0
        self.best_miou = 0.0

        # 检查点目录
        self.checkpoint_dir = Path(train_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        loss_components = {}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.train_config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            # 数据移到设备
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            # 获取特征和logits
            logits, features = self.model(images, return_features=True)

            # 计算损失
            semantic_feat = features.get('semantic') if features else None
            target_semantic = self._get_target_semantic(images) if semantic_feat is not None else None

            loss, loss_dict = self.criterion(logits, labels, semantic_feat, target_semantic)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in loss_components:
                    loss_components[k] = 0.0
                loss_components[k] += v

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        # 平均损失
        avg_loss = total_loss / len(train_loader)
        for k in loss_components:
            loss_components[k] /= len(train_loader)

        return avg_loss, loss_components

    def validate(self, val_loader):
        """验证"""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # 前向传播
                logits = self.model(images)

                # 计算损失
                loss, _ = self.criterion(logits, labels, None, None)
                total_loss += loss.item()

                # 预测
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        # 计算指标
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = self._compute_metrics(all_preds, all_labels)

        return total_loss / len(val_loader), metrics

    def _compute_metrics(self, preds, labels):
        """计算评估指标"""
        # mIoU
        iou_per_class = []
        for cls in range(self.model_config.num_classes):
            pred_mask = (preds == cls)
            label_mask = (labels == cls)

            intersection = (pred_mask & label_mask).sum().float()
            union = (pred_mask | label_mask).sum().float()

            if union > 0:
                iou = intersection / union
            else:
                iou = torch.tensor(0.0)

            iou_per_class.append(iou)

        miou = torch.stack(iou_per_class).mean()

        # 准确率
        accuracy = (preds == labels).float().mean()

        return {
            'miou': miou.item(),
            'accuracy': accuracy.item(),
            'iou_per_class': [iou.item() for iou in iou_per_class]
        }

    def _get_target_semantic(self, images):
        """获取目标语义特征（使用CLIP）"""
        if self.model.clip_branch is not None:
            with torch.no_grad():
                # 调整图像尺寸
                images_resized = torch.nn.functional.interpolate(
                    images, size=(224, 224), mode='bilinear', align_corners=False
                )
                semantic = self.model.clip_branch(images_resized)
            return semantic
        return None

    def train(self, train_loader, val_loader):
        """完整训练流程"""
        print(f"[Training] Starting training for {self.train_config.epochs} epochs")
        print(f"[Training] Device: {self.device}")
        print(f"[Training] Batch size: {self.train_config.batch_size}")

        for epoch in range(1, self.train_config.epochs + 1):
            self.current_epoch = epoch

            # 训练
            train_loss, loss_dict = self.train_epoch(train_loader, epoch)

            # 验证
            if epoch % self.train_config.val_interval == 0:
                val_loss, metrics = self.validate(val_loader)

                # 学习率调度
                self.scheduler.step()

                # 记录到TensorBoard (可选)
                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', train_loss, epoch)
                    self.writer.add_scalar('Loss/val', val_loss, epoch)
                    self.writer.add_scalar('Metrics/mIoU', metrics['miou'], epoch)
                    self.writer.add_scalar('Metrics/Accuracy', metrics['accuracy'], epoch)
                    self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f'LossComponents/{k}', v, epoch)

                # 打印结果
                print(f"\nEpoch {epoch}/{self.train_config.epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  mIoU: {metrics['miou']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")

                # 保存最佳模型
                if metrics['miou'] > self.best_miou:
                    self.best_miou = metrics['miou']
                    self.save_checkpoint(epoch, metrics['miou'], 'best_model.pth')
                    print(f"  [+] Best model saved (mIoU: {self.best_miou:.4f})")

            # 定期保存
            if epoch % self.train_config.save_interval == 0:
                self.save_checkpoint(epoch, 0, f'checkpoint_epoch_{epoch}.pth')

        print(f"\n[Training] Completed! Best mIoU: {self.best_miou:.4f}")

    def save_checkpoint(self, epoch, miou, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'miou': miou,
            'config': {
                'model': self.model_config.__dict__,
                'train': self.train_config.__dict__
            }
        }

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"[Checkpoint] Saved to {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_miou = checkpoint.get('miou', 0.0)

        print(f"[Checkpoint] Loaded from {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}, Best mIoU: {self.best_miou:.4f}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Train Part2 Enhanced Model')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()

    # 加载配置
    model_cfg = get_config('model')
    train_cfg = get_config('train')

    # 命令行参数覆盖
    train_cfg.batch_size = args.batch_size
    train_cfg.epochs = args.epochs
    train_cfg.learning_rate = args.lr
    train_cfg.device = args.device

    # 创建数据加载器
    print("[Data] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_paths=[
            os.path.join(train_cfg.data_root, "USIS10K_Processed/USIS10K_Processed")
        ],
        val_path=os.path.join(train_cfg.data_root, "SUIM_Processed/SUIM_Processed"),
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        image_size=model_cfg.input_size,
        num_classes=model_cfg.num_classes
    )

    # 创建训练器
    trainer = Trainer(model_cfg, train_cfg)

    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # 开始训练
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
