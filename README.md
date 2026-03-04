# 水下图像全景分割系统 (Underwater Image Panoptic Segmentation)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于**CNN + CBAM + CLIP语义约束 + 扩散特征融合**的水下图像全景分割系统，支持轻量化部署到嵌入式平台。

## 项目特性

- **多模态特征融合**: CNN视觉特征 + CLIP语义特征 + 扩散模型特征
- **注意力机制**: CBAM (Channel & Spatial Attention)
- **金字塔池化**: 多尺度特征提取 (PSP)
- **轻量化部署**: INT8量化 + TensorRT优化
- **完整Demo**: CLI + PyQt5界面

## 模型性能

| 指标 | 数值 |
|------|------|
| 参数量 | 19.10M |
| 验证集 mIoU | **84.18%** |
| 验证集 Accuracy | **98.47%** |
| CPU 推理速度 | 788 ms/image |
| Jetson NX 预估 | 78.8 ms/image (12.7 FPS) |

## 模型权重

由于模型文件较大，未包含在本仓库中。请通过以下方式获取：

### 方式1: 训练自己的模型

```bash
cd Part2_Enhanced
python train.py --epochs 50 --batch_size 4
```

### 方式2: 使用预训练模型

| 模型 | 大小 | 下载链接 |
|------|------|----------|
| best_model.pth | 213MB | [百度网盘](链接) / [Google Drive](链接) |
| model_quantized.pth | 57MB | [百度网盘](链接) / [Google Drive](链接) |

下载后放置在对应目录：
```bash
# 训练模型
cp best_model.pth Part2_Enhanced/checkpoints/

# 量化模型
cp model_quantized.pth Part3_Lightweight/results/quantized/
```

## 项目结构

```
.
├── Part2_Enhanced/           # 增强型分割模型
│   ├── configs/              # 模型配置
│   ├── models/               # 模型实现
│   │   ├── backbone.py       # CNN+CBAM+PSP
│   │   ├── clip_branch.py    # CLIP语义分支
│   │   ├── diffusion_branch.py # 扩散特征分支
│   │   ├── fusion.py         # 特征融合模块
│   │   └── seg_model.py      # 完整分割模型
│   ├── data/                 # 数据集加载
│   ├── losses/               # 损失函数
│   ├── eval/                 # 评估工具
│   ├── train.py              # 训练脚本
│   └── evaluate.py           # 评估脚本
│
├── Part3_Deployment_Demo/    # 部署Demo
│   ├── 05_Shared/            # 共享代码
│   │   ├── models/           # 模型包装器
│   │   └── common/           # 通用工具
│   ├── demo_cli.py           # CLI Demo
│   └── main.py               # 主程序
│
├── Part3_Lightweight/        # 轻量化工具
│   ├── 01_quantize/          # INT8量化
│   ├── 02_prune/             # 网络剪枝
│   └── results/              # 轻量化报告
│
├── FINAL_REPORT.md           # 项目最终报告
└── README.md                 # 本文件
```

## 快速开始

### 环境安装

```bash
# 创建虚拟环境 (推荐使用 uv)
pip install uv
uv venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# 安装依赖
pip install torch torchvision numpy pillow matplotlib tqdm opencv-python scikit-learn einops
```

## 所有可用命令

### Part2: 模型训练与评估

```bash
# 进入Part2目录
cd Part2_Enhanced

# 训练模型 (使用SUIM + USIS10K数据集)
python train.py

# 评估训练好的模型
python evaluate.py

# 测试模型配置
python configs/model_config.py
```

### Part3: 部署与可视化

```bash
# 进入Demo目录
cd Part3_Deployment_Demo

# CLI Demo - 生成可视化对比图
python demo_cli.py

# 生成所有图表 (柱状图、热力图、灵敏度曲线、可视化对比)
python generate_all_charts.py

# PyQt5 GUI Demo (图形界面)
python main.py gui

# 运行性能测试
python main.py benchmark

# 嵌入式部署模拟
python main.py simulate
```

### Part3: 模型轻量化

```bash
# 进入轻量化目录
cd Part3_Lightweight

# 生成轻量化报告 (量化、剪枝)
python generate_report.py

# 运行完整轻量化流程
python run_lightweight.py
```

### Part1: 数据集处理

```bash
# 处理SUIM数据集
cd Part1_DataProcessing
python process_suim.py --input "path/to/SUIM" --output "SUIM_Processed"

# 处理USIS10K数据集
python process_usis10k.py --input "path/to/USIS10K" --output "USIS10K_Processed"

# 处理UIIS10K数据集
python process_uiis10k.py --input "path/to/UIIS10K" --output "UIIS10K_Processed"
```

## 数据集

| 数据集 | 用途 | 数量 |
|--------|------|------|
| SUIM | 训练+验证 | 7200张 |
| USIS10K | 训练 | 3122张 |
| UIIS10K | 测试 | 10050张 |

### 数据集类别

| ID | 类别名称 | 颜色 |
|----|----------|------|
| 0 | Background (水体) | 黑色 |
| 1 | Human divers | 红色 |
| 2 | Plants and sea grass | 绿色 |
| 3 | Wrecks and ruins | 蓝色 |
| 4 | Robots (AUVs/ROVs) | 黄色 |
| 5 | Reefs and invertebrates | 品红 |
| 6 | Fish and vertebrates | 青色 |
| 7 | Sea floor and rocks | 灰色 |

## 模型架构

```
Input [3, 256, 256]
    ↓
┌─────────────────────────────────────┐
│     Multi-Branch Encoder            │
├─────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐       │
│  │   CNN    │  │  CLIP    │       │
│  │  +CBAM   │  │ Semantic │       │
│  │   +PSP   │  │  Branch  │       │
│  └──────────┘  └──────────┘       │
│       │              │             │
│       └──────┬───────┘             │
│              ↓                     │
│      ┌──────────────┐             │
│      │  Diffusion   │             │
│      │   Encoder    │             │
│      └──────────────┘             │
└──────────────────┼────────────────┘
                   ↓
          ┌─────────────────┐
          │ Fusion Module   │
          │ (Multi-head Attn)│
          └─────────────────┘
                   ↓
          ┌─────────────────┐
          │ Segmentation    │
          │ Head (8 classes)│
          └─────────────────┘
                   ↓
              Output [8, H, W]
```

## 详细命令参考

### Part2_Enhanced 命令

| 命令 | 说明 |
|------|------|
| `python train.py` | 训练分割模型 (50 epochs, batch_size=4) |
| `python evaluate.py` | 评估模型并生成报告和可视化 |
| `python configs/model_config.py` | 测试模型配置 |

**输出文件**:
- `checkpoints/best_model.pth` - 最佳模型权重
- `logs/` - 训练日志
- `eval_results/` - 评估结果 (包含可视化对比图)

### Part3_Deployment_Demo 命令

| 命令 | 说明 |
|------|------|
| `python demo_cli.py` | CLI Demo: 生成图像分割对比图 |
| `python generate_all_charts.py` | 生成所有分析图表和报告 |
| `python main.py gui` | PyQt5 图形界面Demo |
| `python main.py benchmark` | 运行性能基准测试 |
| `python main.py simulate` | 嵌入式部署模拟 |

**输出文件**:
- `output/charts/` - 模型对比柱状图、误差热力图、灵敏度曲线
- `output/comparisons/` - 原始图像 vs 分割结果对比图
- `output/experiment_report.md` - 完整实验报告

### Part3_Lightweight 命令

| 命令 | 说明 |
|------|------|
| `python generate_report.py` | 生成轻量化报告 (量化、剪枝分析) |
| `python run_lightweight.py` | 运行完整轻量化流程 |

**输出文件**:
- `results/LIGHTWEIGHT_REPORT.md` - 模型优化报告
- `results/quantized/` - 量化后的模型

### part2 命令 (原始模型)

| 命令 | 说明 |
|------|------|
| `python main.py` | 训练基础 CNN+CBAM 模型 |

**输出文件**:
- `best_model.pth` - 模型权重 (213MB)

## 部署指南

### Jetson Xavier NX 部署

```bash
# 1. 导出ONNX模型
python export_onnx.py --checkpoint best_model.pth

# 2. 转换为TensorRT引擎 (在Jetson上)
trtexec --onnx=model.onnx --saveEngine=model.trt --int8

# 3. 运行推理
python infer_tensorrt.py --engine model.trt --input image.jpg
```

### 性能预估

| 平台 | 推理时间 | FPS |
|------|----------|-----|
| CPU (x86) | 788 ms | 1.3 |
| Jetson NX GPU | 78.8 ms | 12.7 |
