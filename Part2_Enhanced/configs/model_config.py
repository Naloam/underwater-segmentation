"""
Part2 Enhanced - Model Configuration

轻量化全景分割模型配置
- Backbone: CNN + CBAM
- CLIP语义分支: HuggingFace预训练
- 扩散特征分支: 简化U-Net编码器
- 融合模块: 轻量级多头注意力
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础配置
    num_classes: int = 8
    input_size: Tuple[int, int] = (256, 256)

    # Backbone配置
    backbone_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    use_cbam: bool = True
    cbam_reduction: int = 16
    use_pyramid_pool: bool = True
    pool_scales: Tuple[int, ...] = (1, 2, 3, 6)

    # CLIP配置
    use_clip: bool = True
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_embed_dim: int = 512
    clip_out_dim: int = 256
    freeze_clip: bool = True

    # 扩散特征分支配置
    use_diffusion: bool = True
    diffusion_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    diffusion_out_dim: int = 256

    # 融合模块配置
    fusion_dim: int = 256
    num_heads: int = 4
    fusion_dropout: float = 0.1

    # 损失函数权重
    ce_loss_weight: float = 1.0
    semantic_loss_weight: float = 1.0
    pq_loss_weight: float = 2.0


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据路径
    data_root: str = r"D:\myProjects\大创(1)"
    train_datasets: List[str] = field(default_factory=lambda: [
        "USIS10K_Processed/USIS10K_Processed"
    ])
    val_dataset: str = "SUIM_Processed/SUIM_Processed"
    test_dataset: str = "UIIS10K_Processed/UIIS10K_Processed"

    # 训练参数
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # 特征融合权重
    clip_weight: float = 0.3
    diffusion_weight: float = 0.3

    # 保存设置
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_interval: int = 5
    val_interval: int = 1

    # 设备
    device: str = "cuda"  # 自动检测

    def __post_init__(self):
        """自动检测设备"""
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class AblationConfig:
    """消融实验配置"""
    experiments: List[dict] = field(default_factory=lambda: [
        {"name": "baseline", "use_clip": False, "use_diffusion": False, "use_fusion": False},
        {"name": "with_clip", "use_clip": True, "use_diffusion": False, "use_fusion": True},
        {"name": "with_diffusion", "use_clip": False, "use_diffusion": True, "use_fusion": True},
        {"name": "full", "use_clip": True, "use_diffusion": True, "use_fusion": True},
    ])


# 全局配置实例
model_cfg = ModelConfig()
train_cfg = TrainingConfig()
ablation_cfg = AblationConfig()


def get_config(name: str = "model"):
    """获取配置"""
    configs = {
        "model": model_cfg,
        "train": train_cfg,
        "ablation": ablation_cfg,
    }
    return configs.get(name)


if __name__ == "__main__":
    print("Model Config:")
    print(f"  Num Classes: {model_cfg.num_classes}")
    print(f"  Input Size: {model_cfg.input_size}")
    print(f"  Use CLIP: {model_cfg.use_clip}")
    print(f"  Use Diffusion: {model_cfg.use_diffusion}")
    print(f"  CBAM Reduction: {model_cfg.cbam_reduction}")

    print("\nTraining Config:")
    print(f"  Batch Size: {train_cfg.batch_size}")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Learning Rate: {train_cfg.learning_rate}")
    print(f"  Device: {train_cfg.device}")

    print("\nAblation Experiments:")
    for exp in ablation_cfg.experiments:
        print(f"  - {exp['name']}: CLIP={exp['use_clip']}, Diffusion={exp['use_diffusion']}, Fusion={exp['use_fusion']}")
