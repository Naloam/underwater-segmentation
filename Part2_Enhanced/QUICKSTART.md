# Part2 Enhanced - 快速开始指南

## 当前状态 ✅

- [x] Day 1-2: 项目结构和配置文件
- [x] Day 3-4: 核心模块实现
- [x] Day 5-6: 损失函数和训练框架
- [x] 基础模型测试通过 (5.42M参数，不含CLIP/Diffusion)

## 快速开始

### 1. 安装依赖 (可选: CLIP)

```bash
cd d:\myProjects\大创(1)\Part2_Enhanced

# 基础依赖
pip install torch torchvision pillow numpy tqdm scikit-image matplotlib seaborn pyyaml

# CLIP依赖 (可选，用于启用语义分支)
pip install transformers einops

# 验证安装
python -c "import torch; print('PyTorch:', torch.__version__)"
```

### 2. 测试模型

```bash
# 测试基础模型 (无需CLIP)
python -c "
from models import create_model
from configs.model_config import ModelConfig
import torch

config = ModelConfig(use_clip=False, use_diffusion=False)
model = create_model(config)

x = torch.randn(2, 3, 256, 256)
logits = model(x)
print(f'Output: {logits.shape}')
"
```

### 3. 准备数据

确保数据集路径正确：
```
d:\myProjects\大创(1)\
├── SUIM_Processed\SUIM_Processed\1_raw\
├── SUIM_Processed\SUIM_Processed\6_label\
├── USIS10K_Processed\USIS10K_Processed\1_raw\
└── ...
```

### 4. 开始训练

```bash
# 基础训练 (无CLIP/Diffusion)
python train.py --batch-size 4 --epochs 50 --device cuda

# 或者修改 configs/model_config.py 中的数据路径
```

## 模型配置

### Baseline (当前)
- 参数量: 5.42M (全部可训练)
- 包含: CNN + CBAM + 金字塔池化
- mIoU目标: ~82-85%

### Full (需要transformers)
- 参数量: ~165M (10M可训练 + 155M冻结CLIP)
- 包含: + CLIP语义分支 + 扩散特征分支
- mIoU目标: ~86-88%

## 训练计划

```bash
# Stage 1: 基础训练 (2-3天)
python train.py --batch-size 4 --epochs 50

# Stage 2: 安装CLIP后，启用语义分支
# 修改 configs/model_config.py: use_clip=True
python train.py --batch-size 4 --epochs 20 --lr 5e-5 --resume checkpoints/best_model.pth
```

## 输出文件

```
Part2_Enhanced/
├── checkpoints/
│   ├── best_model.pth          # 最佳模型
│   └── checkpoint_epoch_*.pth   # 定期检查点
└── logs/
│   └── 20260303_XXXXXX/        # TensorBoard日志
```

## 监控训练

```bash
# TensorBoard
tensorboard --logdir logs/
```

## 已知问题

1. **transformers未安装**: CLIP分支会使用fallback (随机投影)
   - 解决: `pip install transformers`
   - 或设置 `config.use_clip=False`

2. **BatchNorm需要batch_size > 1**: 测试时batch_size=1会报错
   - 解决: 使用 `batch_size >= 2`

3. **显存不足**: 降低batch_size或禁用部分分支
   - 解决: `--batch-size 2` 或设置 `use_clip=False, use_diffusion=False`

## 下一步

1. 安装完整依赖: `pip install -r requirements.txt`
2. 准备数据集
3. 开始训练: `python train.py`
4. 查看TensorBoard监控训练

---

*项目框架已完成，可以开始训练！*
