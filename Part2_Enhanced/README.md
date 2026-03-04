# Part2 Enhanced - 轻量化全景分割模型

## 项目概述

基于混合方案的水下图像全景分割模型，整合了CNN+CBAM backbone、CLIP语义特征和扩散特征。

## 架构设计

```
轻量化全景分割模型
├── Backbone: CNN + CBAM + 金字塔池化
├── CLIP语义分支: HuggingFace预训练 (冻结权重)
├── 扩散特征分支: 简化U-Net编码器
└── 融合模块: 轻量级多头注意力
```

## 项目结构

```
Part2_Enhanced/
├── configs/
│   └── model_config.py          # 模型和训练配置
├── models/
│   ├── backbone.py              # 增强Backbone (CBAM + 金字塔池化)
│   ├── clip_branch.py           # CLIP语义分支
│   ├── diffusion_branch.py      # 扩散特征分支
│   ├── fusion.py                # 特征融合模块
│   └── seg_model.py             # 完整分割模型
├── losses/
│   ├── semantic_loss.py         # 语义匹配损失
│   └── pq_loss.py               # PQ损失 + 组合损失
├── data/
│   └── dataset.py               # 数据集加载
├── eval/
│   ├── metrics.py               # 评估指标
│   └── visualize.py             # 可视化工具
├── train.py                      # 训练脚本
├── requirements.txt             # 依赖列表
└── README.md                    # 本文件
```

## 核心特性

### 1. 增强Backbone
- 复用Part2的CNN架构
- CBAM注意力模块 (通道 + 空间)
- 金字塔池化模块 (多尺度特征)
- 4个stage: [64, 128, 256, 512]通道

### 2. CLIP语义分支
- 使用HuggingFace预训练CLIP模型
- 权重冻结，仅作为特征提取器
- 自动延迟加载，内存友好

### 3. 扩散特征分支
- 简化U-Net编码器 (非完整扩散模型)
- 多尺度特征提取
- 可选自注意力模块

### 4. 轻量级融合
- 多头注意力融合
- 语义门控机制
- 多尺度特征融合

### 5. 损失函数
- CE Loss (基础)
- Focal Loss (处理不平衡)
- Dice Loss (对小目标敏感)
- PQ Loss (全景分割质量)
- Semantic Match Loss (语义一致性)

## 环境搭建

```bash
# 创建conda环境
conda create -n part2_env python=3.9 -y
conda activate part2_env

# 安装依赖
cd Part2_Enhanced
pip install -r requirements.txt

# (可选) 如果有GPU，安装CUDA版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 快速开始

### 1. 测试模型

```bash
# 测试模型创建
python -m models.seg_model

# 测试Backbone
python -m models.backbone

# 测试CLIP分支
python -m models.clip_branch

# 测试融合模块
python -m models.fusion
```

### 2. 训练模型

```bash
# 基础训练 (batch_size=4, epochs=50)
python train.py --batch-size 4 --epochs 50 --device cuda

# 从检查点恢复
python train.py --resume checkpoints/best_model.pth

# 自定义参数
python train.py --batch-size 2 --epochs 100 --lr 5e-5
```

### 3. 评估模型

```python
from models import create_model
from eval.metrics import compute_metrics
from eval.visualize import visualize_prediction
import torch

# 加载模型
model = create_model()
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    logits = model(images)
    preds = torch.argmax(logits, dim=1)

# 计算指标
metrics = compute_metrics(preds, targets, num_classes=8)
print(f"mIoU: {metrics['miou']:.4f}")
```

## 配置说明

### ModelConfig
```python
- num_classes: 8                    # 类别数
- input_size: (256, 256)            # 输入尺寸
- backbone_channels: [64,128,256,512]
- use_cbam: True                    # 使用CBAM
- use_pyramid_pool: True            # 使用金字塔池化
- use_clip: True                    # 使用CLIP分支
- use_diffusion: True               # 使用扩散分支
```

### TrainingConfig
```python
- batch_size: 4                     # 批次大小
- epochs: 50                        # 训练轮数
- learning_rate: 1e-4               # 学习率
- ce_loss_weight: 1.0               # CE损失权重
- pq_loss_weight: 2.0               # PQ损失权重
- semantic_loss_weight: 1.0         # 语义损失权重
```
