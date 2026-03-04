# Part3_Deployment_Demo

水下图像增强 + 全景分割系统 - 第三部分：模型轻量化部署与Demo开发

## 项目概述

本项目是大创项目的第三部分，负责将第二部分训练好的模型进行轻量化部署，并开发可演示的场景化Demo系统。

**核心功能：**
- 水下图像增强
- 全景分割
- 模型轻量化（知识蒸馏、网络剪枝、INT8量化）
- 嵌入式部署（Jetson Xavier NX）
- PyQt5桌面端Demo

## 项目结构

```
Part3_Deployment_Demo/
├── 01_DataVisualization/       # 数据可视化模块
├── 02_ModelOptimization/       # 模型轻量化模块
├── 03_EmbeddedDeployment/      # 嵌入式部署模块
├── 04_Demo/                    # PyQt5 Demo应用
├── 05_Shared/                  # 共享模块
│   ├── models/                 # Mock模型和统一接口
│   └── common/                 # 通用工具
├── requirements.txt            # 依赖清单
└── README.md                   # 本文件
```

## 快速开始

### 环境要求

- Python 3.9+
- CUDA 11.8+ (GPU加速)

### 安装依赖

```bash
cd Part3_Deployment_Demo
pip install -r requirements.txt
```

### 运行Demo

```bash
cd 04_Demo/src
python demo_app.py
```

### 生成可视化对比图

```bash
cd 01_DataVisualization/src
python visual_comparison.py
```

### 运行嵌入式部署模拟

```bash
cd 03_EmbeddedDeployment/src
python deployment_simulator.py
```

## 模块说明

### 01_DataVisualization

数据可视化模块，用于生成实验结果图表。

**主要功能：**
- 生成 原始→增强→分割 的对比图
- 生成柱状图、热力图、折线图等实验图表
- 生成灵敏度分析曲线
- 生成实验报告

**使用示例：**
```python
from src.visual_comparison import VisualComparisonGenerator

generator = VisualComparisonGenerator("path/to/SUIM_Processed")
generator.batch_generate("output/dir", num_samples=10)
```

### 02_ModelOptimization

模型轻量化模块，实现模型压缩和加速。

**主要技术：**
- 知识蒸馏：将Mask2Former蒸馏到轻量级模型
- 网络剪枝：去除不重要的通道
- INT8量化：进一步压缩模型体积

**目标指标：**
- 模型体积：≤500MB
- 推理时间：≤1秒（CPU）
- 精度损失：<3% mIoU

### 03_EmbeddedDeployment

嵌入式部署模块，支持Jetson Xavier NX平台。

**主要功能：**
- TensorRT模型转换
- 性能模拟测试（无硬件时使用）
- 资源监控（FPS、功耗、温度）

**使用示例：**
```python
from src.deployment_simulator import JetsonSimulator

simulator = JetsonSimulator()
report = simulator.generate_report(model_info)
print(report)
```

### 04_Demo

PyQt5桌面端Demo应用。

**主要功能：**
- 图像/视频输入
- 实时增强和分割
- 多场景预设（浅海珊瑚礁、深海遗迹、海洋生物监测）
- 结果保存

**场景预设：**
- 浅海珊瑚礁：光线充足，色彩丰富
- 深海遗迹：低光照，高散射
- 海洋生物监测：动态场景，高帧率

### 05_Shared

共享模块，包含Mock模型和通用工具。

**Mock模型：**
在第二部分模型训练完成前，使用Mock模型进行前端开发。

**统一接口：**
确保Mock模型和真实模型可以无缝切换。

## 依赖关系

```
第一部分（数据集）─────┐
                       ├──▶ 第二部分（模型训练）──▶ 第三部分（本模块）
                       │
第二部分设计文档 ──────┘
```

**当前状态：**
- ✅ 第一部分数据集已提供
- ⏳ 第二部分模型正在训练中
- ✅ 本模块使用Mock模型并行开发

## 开发状态

| 模块 | 状态 | 说明 |
|------|------|------|
| 共享模块 | ✅ 完成 | Mock模型、统一接口、工具函数 |
| 数据可视化 | ✅ 完成 | 对比图生成、图表生成 |
| PyQt5 Demo | ✅ 完成 | 主窗口UI、图像处理 |
| 嵌入式部署 | ✅ 完成 | 模拟器、性能估算 |
| 模型轻量化 | 🟡 框架完成 | 等待真实模型进行测试 |

## 配置说明

模型切换配置（位于 `05_Shared/common/config_loader.py`）：

```python
model_config = {
    "use_mock": True,  # False时使用真实模型
    "real_model_path": "../扩散模型+CLIP语义约束+全景分割/checkpoints/best_model.pth"
}
```

## 常见问题

### Q: 如何切换到真实模型？

A: 修改配置文件中的 `use_mock` 为 `False`，并指定正确的模型路径。

### Q: 没有Jetson硬件如何测试？

A: 使用内置的模拟器进行理论性能估算，预留真实硬件测试接口。

### Q: Demo显示图像异常？

A: 检查OpenCV的BGR/RGB转换，确保图像格式正确。

## 文献引用

本项目使用的模型和技术：

- Mask2Former: [论文链接]
- SUIM数据集: [数据集链接]
- USIS10K数据集: [数据集链接]

## 联系方式

- 项目负责人：第三部分同学
- 技术支持：项目组

## 许可证

本项目仅供学术研究使用。

---

*最后更新：2025-03-02*
