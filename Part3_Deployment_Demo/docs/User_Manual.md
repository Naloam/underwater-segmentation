# Part3_Deployment_Demo 项目文档

## 项目概述

第三部分：模型轻量化部署 + Demo开发

## 目录结构

```
Part3_Deployment_Demo/
├── 01_DataVisualization/       # 数据可视化模块
├── 02_ModelOptimization/       # 模型轻量化模块
├── 03_EmbeddedDeployment/      # 嵌入式部署模块
├── 04_Demo/                    # PyQt5 Demo应用
├── 05_Shared/                  # 共享模块
├── main.py                     # 主入口脚本
├── requirements.txt            # 依赖清单
└── README.md                   # 项目说明
```

## 模块说明

### 01_DataVisualization

数据可视化模块，用于生成实验结果图表和报告。

- **visual_comparison.py**: 生成原始→增强→分割对比图
- **chart_generator.py**: 生成柱状图、热力图、折线图等
- **report_generator.py**: 生成Markdown/HTML/JSON格式报告

### 02_ModelOptimization

模型轻量化模块，实现模型压缩和加速。

- **knowledge_distillation/**: 知识蒸馏
- **pruning/**: 网络剪枝
- **quantization/**: INT8量化
- **optimization_pipeline.py**: 综合优化流水线

### 03_EmbeddedDeployment

嵌入式部署模块，支持Jetson Xavier NX平台。

- **deployment_simulator.py**: Jetson Xavier NX性能模拟器

### 04_Demo

PyQt5桌面端Demo应用。

- **ui/main_window.py**: 主窗口界面
- **core/inference_engine.py**: 推理引擎
- **communication/**: 串口/网络通信接口
- **resources/scenarios/**: 场景配置文件
- **resources/styles/**: UI样式

### 05_Shared

共享模块，包含Mock模型和通用工具。

- **models/mock_models.py**: Mock模型（开发阶段使用）
- **models/model_interface.py**: 统一模型接口
- **common/utils.py**: 通用工具函数
- **common/config_loader.py**: 配置加载器

## 使用方法

### 启动Demo

```bash
python main.py demo
```

### 生成可视化

```bash
python main.py visualize
```

### 生成报告

```bash
python main.py report
```

### 运行部署模拟

```bash
python main.py simulate
```

### 运行测试

```bash
python main.py test
```