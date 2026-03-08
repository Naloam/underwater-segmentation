"""
Evaluation Script

评估训练好的模型并生成可视化结果
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from configs.model_config import ModelConfig
from data import create_dataloaders
from eval.metrics import compute_metrics, MetricTracker
from eval.visualize import (
    visualize_prediction, save_comparison,
    label_to_color, overlay_mask, CLASS_NAMES, CLASS_COLORS
)
import matplotlib.pyplot as plt


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """加载训练好的模型"""
    print(f"[Eval] Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 检测checkpoint格式
    if 'model_state_dict' in checkpoint:
        print("[Eval] Detected Enhanced Model format")
        if 'config' in checkpoint:
            cfg_dict = checkpoint['config']['model']
            config = ModelConfig(**cfg_dict)
        else:
            config = ModelConfig()
        model = create_model(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        checkpoint_info = checkpoint
    elif 'encoder.0.weight' in checkpoint or any(k.startswith('encoder') for k in checkpoint.keys()):
        print("[Eval] Detected SegModel format (simple CNN+CBAM)")
        model = SegModel(num_classes=8).to(device)
        model.load_state_dict(checkpoint, strict=False)
        config = ModelConfig()
        checkpoint_info = {'epoch': 'N/A', 'miou': 'N/A'}
    else:
        print("[Eval] Unknown format, treating as simple state dict")
        model = SegModel(num_classes=8).to(device)
        model.load_state_dict(checkpoint, strict=False)
        config = ModelConfig()
        checkpoint_info = {'epoch': 'N/A', 'miou': 'N/A'}

    model.eval()

    print(f"[Eval] Model loaded. Epochs: {checkpoint_info.get('epoch', 'N/A')}")
    if checkpoint_info.get('miou', 'N/A') != 'N/A':
        print(f"[Eval] Checkpoint mIoU: {checkpoint_info.get('miou')}")

    return model, config, checkpoint_info


# SegModel from part2 (simple CNN+CBAM)
class SegModel(torch.nn.Module):
    """简单的分割模型 (来自part2)"""
    def __init__(self, num_classes=8):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.ReLU()
        )
        self.decoder = torch.nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        feat = self.encoder(x)
        return self.decoder(feat)


def evaluate(model, dataloader, device: str = 'cpu', num_samples: int = None):
    """评估模型"""
    model.eval()

    tracker = MetricTracker(num_classes=8)
    all_images = []
    all_targets = []
    all_preds = []

    total_batches = len(dataloader)
    if num_samples:
        total_batches = min(total_batches, num_samples // dataloader.batch_size + 1)

    print(f"[Eval] Evaluating {total_batches} batches...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches)):
            if batch_idx >= total_batches:
                break

            images = batch['image'].to(device)
            targets = batch['label'].to(device)

            # 前向传播
            logits = model(images)
            preds = logits.argmax(dim=1)

            # 更新指标
            tracker.update(preds.cpu(), targets.cpu())

            # 保存用于可视化的样本 (最多保存20个)
            if len(all_images) < 20:
                all_images.append(images.cpu())
                all_targets.append(targets.cpu())
                all_preds.append(preds.cpu())

    # 计算平均指标
    metrics = tracker.get_average()

    return metrics, all_images, all_targets, all_preds


def generate_report(
    metrics: dict,
    checkpoint_info: dict,
    output_dir: Path
):
    """生成评估报告"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "evaluation_report.md"

    epoch = checkpoint_info.get('epoch', 'N/A')
    use_clip = checkpoint_info.get('config', {}).get('model', {}).get('use_clip', False)
    use_diffusion = checkpoint_info.get('config', {}).get('model', {}).get('use_diffusion', False)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Part2 Enhanced Model - Evaluation Report\n\n")
        f.write(f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Training Summary\n\n")
        f.write(f"- **Epochs Completed**: {epoch}\n")
        f.write(f"- **Architecture**: CNN + CBAM + Pyramid Pooling + FPN Decoder\n")
        if use_clip:
            f.write(f"- **CLIP Branch**: Enabled (frozen, openai/clip-vit-base-patch32)\n")
        if use_diffusion:
            f.write(f"- **Diffusion Branch**: Enabled\n")

        f.write("\n## Evaluation Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| **mIoU** | **{metrics['miou']:.4f}** |\n")
        f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
        f.write(f"| F1 Score | {metrics['f1']:.4f} |\n")

        f.write("\n## Per-Class IoU\n\n")
        f.write("| Class | IoU |\n")
        f.write("|-------|-----|\n")
        for i, name in enumerate(CLASS_NAMES):
            iou = metrics.get(f'iou_class_{i}', 0.0)
            f.write(f"| {name} | {iou:.4f} |\n")

        f.write("\n## Notes\n\n")
        f.write("- **Training Dataset**: USIS10K (7442 samples)\n")
        f.write("- **Validation Dataset**: SUIM (1440 samples)\n")
        f.write("- **Input Size**: 256x256\n")
        f.write("- **Number of Classes**: 8\n")

    print(f"[Eval] Report saved to {report_path}")

    return report_path


def visualize_results(
    images: list,
    targets: list,
    preds: list,
    output_dir: Path
):
    """生成可视化结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Vis] Generating visualizations...")

    # 保存对比图
    vis_dir = output_dir / "comparisons"
    save_comparison(images, targets, preds, vis_dir)

    # 创建图例
    legend_path = output_dir / "class_legend.png"
    from eval.visualize import create_legend
    fig = create_legend()
    fig.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[Vis] Visualizations saved to {output_dir}")


def create_metrics_plot(
    checkpoint_info: dict,
    output_dir: Path
):
    """创建训练指标图表（需要训练日志数据）"""
    # 如果没有真实训练日志，不生成假图
    print("[Vis] Skipping training curve (no training log available)")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Part2 Enhanced Model')
    parser.add_argument('--checkpoint', type=str, default="output/checkpoint_epoch_20.pth",
                        help='Path to checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device (auto-detect if not set)')
    parser.add_argument('--num-samples', type=int, default=None, help='Limit evaluation samples')
    args = parser.parse_args()

    # 配置
    checkpoint_path = args.checkpoint
    output_dir = Path("eval_results")

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Part2 Enhanced - Model Evaluation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("=" * 60)

    # 加载模型
    model, config, checkpoint = load_model(checkpoint_path, device)

    # 创建数据加载器
    print("\n[Eval] Loading validation data...")
    from configs.model_config import TrainingConfig
    train_cfg = TrainingConfig()

    _, val_loader = create_dataloaders(
        train_paths=[os.path.join(train_cfg.data_root, ds) for ds in train_cfg.train_datasets],
        val_path=os.path.join(train_cfg.data_root, train_cfg.val_dataset),
        batch_size=2,
        num_workers=0,
        num_classes=8
    )

    # 评估模型
    print("\n" + "=" * 60)
    print("Running Evaluation...")
    print("=" * 60)

    metrics, images, targets, preds = evaluate(
        model, val_loader, device, num_samples=args.num_samples
    )

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"mIoU: {metrics['miou']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

    # 打印每类IoU
    for i, name in enumerate(CLASS_NAMES):
        iou = metrics.get(f'iou_class_{i}', 0.0)
        print(f"  {name}: {iou:.4f}")

    # 生成报告
    print("\n[Eval] Generating report...")
    report_path = generate_report(metrics, checkpoint, output_dir)

    # 生成可视化
    print("\n[Eval] Generating visualizations...")
    visualize_results(images, targets, preds, output_dir)

    # 保存指标JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"- Report: {report_path.name}")
    print(f"- Metrics: {metrics_path.name}")
    print(f"- Visualizations: {output_dir / 'comparisons'}")


if __name__ == "__main__":
    main()
