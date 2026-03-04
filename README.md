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

### 训练模型

```bash
cd Part2_Enhanced
python train.py --epochs 50 --batch_size 4
```

### 评估模型

```bash
cd Part2_Enhanced
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### 运行Demo

```bash
cd Part3_Deployment_Demo
python demo_cli.py
```

### 模型轻量化

```bash
cd Part3_Lightweight
python generate_report.py
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

## 贡献者

- 项目团队: 大创项目组
- 指导教师: [待补充]

## 许可证

MIT License

## 致谢

- SUIM Dataset: [https://github.com/Xiaoyu-BMI/SUIM-Net](https://github.com/Xiaoyu-BMI/SUIM-Net)
- USIS10K Dataset: [https://github.com/HzFu/Underwater_Semantic_Segmentation](https://github.com/HzFu/Underwater_Semantic_Segmentation)
- CLIP: OpenAI

## 联系方式

- 项目地址: [GitHub URL]
- 问题反馈: [Issues]
