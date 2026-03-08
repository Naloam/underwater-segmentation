"""
消融实验脚本

加载 epoch_20 checkpoint，分别用4种配置做 forward 评估:
  baseline:        use_clip=False, use_diffusion=False
  with_clip:       use_clip=True,  use_diffusion=False
  with_diffusion:  use_clip=False, use_diffusion=True
  full:            use_clip=True,  use_diffusion=True
"""

import os, sys, json, torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from configs.model_config import ModelConfig, TrainingConfig, AblationConfig
from data import create_dataloaders
from eval.metrics import MetricTracker
from eval.visualize import CLASS_NAMES


def run_ablation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("output/checkpoint_epoch_20.pth")

    print(f"[Ablation] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Dataloader (val only)
    train_cfg = TrainingConfig()
    _, val_loader = create_dataloaders(
        train_paths=[os.path.join(train_cfg.data_root, ds) for ds in train_cfg.train_datasets],
        val_path=os.path.join(train_cfg.data_root, train_cfg.val_dataset),
        batch_size=2, num_workers=0, num_classes=8,
    )

    ablation_cfg = AblationConfig()
    results = {}

    for exp in ablation_cfg.experiments:
        name = exp["name"]
        print(f"\n{'='*60}")
        print(f"[Ablation] Experiment: {name}")
        print(f"  use_clip={exp['use_clip']}, use_diffusion={exp['use_diffusion']}")
        print("=" * 60)

        cfg = ModelConfig(use_clip=exp["use_clip"], use_diffusion=exp["use_diffusion"])
        model = create_model(cfg).to(device)

        # Load compatible weights (strict=False skips mismatched keys)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Missing keys:    {len(missing)}")
        print(f"  Unexpected keys: {len(unexpected)}")

        model.eval()
        tracker = MetricTracker(num_classes=8)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=name):
                images = batch["image"].to(device)
                targets = batch["label"]
                logits = model(images)
                preds = logits.argmax(dim=1).cpu()
                tracker.update(preds, targets)

        metrics = tracker.get_average()
        results[name] = metrics

        print(f"\n  mIoU:     {metrics['miou']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
        for i, cname in enumerate(CLASS_NAMES):
            print(f"    {cname}: {metrics.get(f'iou_class_{i}', 0):.4f}")

    # Save results
    out_path = Path("eval_results/ablation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Ablation] Results saved to {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("Ablation Summary")
    print("=" * 60)
    print(f"{'Experiment':<20} {'mIoU':>8} {'Acc':>8} {'F1':>8}")
    print("-" * 50)
    for name, m in results.items():
        print(f"{name:<20} {m['miou']:>8.4f} {m['accuracy']:>8.4f} {m['f1']:>8.4f}")

    return results


if __name__ == "__main__":
    run_ablation()
