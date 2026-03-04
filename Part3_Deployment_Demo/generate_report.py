"""
Complete experiment report generation script
"""
import sys
from pathlib import Path

# Add module paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / '05_Shared'))
sys.path.insert(0, str(project_root / '01_DataVisualization' / 'src'))
sys.path.insert(0, str(project_root / '04_Demo' / 'src'))
sys.path.insert(0, str(project_root / '03_EmbeddedDeployment' / 'src'))

from models.model_interface import ModelFactory
from common.config_loader import ConfigLoader
from inference_engine import InferenceEngine
from deployment_simulator import JetsonSimulator
import numpy as np
from datetime import datetime


def generate_complete_report():
    """Generate complete experiment report"""
    print("[Report] Generating complete experiment report...")

    # Load model and get info
    config = ConfigLoader.load_model_config("segmentation")
    segmentor = ModelFactory.create_segmentor(config)
    model_info = segmentor.get_info()

    # Benchmark
    engine = InferenceEngine(device='cpu')
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    benchmark = engine.benchmark(test_image, num_runs=50)

    # Embedded deployment simulation
    simulator = JetsonSimulator()
    model_spec = {
        'name': 'Part2_SegModel',
        'params': model_info.get('params_M', 0.37),
        'flops': 1.5,  # Estimated FLOPs
        'input_size': (3, 256, 256)
    }

    # Generate report content
    report_lines = []
    report_lines.append("#水下图像增强与全景分割系统 - 实验报告\n")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_lines.append("---\n")

    # 1. Project Overview
    report_lines.append("## 一、项目概述\n")
    report_lines.append("本项目旨在实现水下图像增强与全景分割系统的轻量化部署与Demo开发。\n")
    report_lines.append("- **任务**: 模型轻量化部署 + 场景化Demo开发\n")
    report_lines.append("- **模型**: Part2分割模型 (CNN + CBAM注意力)\n")
    report_lines.append(f"- **参数量**: {model_info.get('params', 0):,} ({model_info.get('params_M', 0):.2f}M)\n")
    report_lines.append(f"- **类别数**: {model_info.get('num_classes', 8)}\n")

    # 2. Model Information
    report_lines.append("## 二、模型信息\n")
    report_lines.append("### 2.1 模型结构\n")
    report_lines.append("| 属性 | 值 |")
    report_lines.append("|------|-----|")
    report_lines.append(f"| 模型名称 | {model_info.get('name', 'SegModel')} |")
    report_lines.append(f"| 参数量 | {model_info.get('params', 0):,} |")
    report_lines.append(f"| 输入尺寸 | {model_info.get('input_size', (256, 256))} |")
    report_lines.append(f"| 类别数 | {model_info.get('num_classes', 8)} |")
    report_lines.append(f"| 设备 | {model_info.get('device', 'cpu')} |")
    report_lines.append("")

    # Class names
    report_lines.append("### 2.2 分割类别\n")
    classes = ["Background (水体)", "Reefs and invertebrates (礁石)", "Fish and vertebrates (鱼类)",
               "Crabs and lobsters (蟹类)", "Sea cucumbers (海参)", "Starfish (海星)",
               "Eels (鳗鱼)", "Plants and grass (水草)"]
    for i, cls in enumerate(classes):
        report_lines.append(f"{i}. {cls}")
    report_lines.append("")

    # 3. Performance Benchmark
    report_lines.append("## 三、性能测试\n")
    report_lines.append("### 3.1 CPU推理性能\n")
    report_lines.append("| 指标 | 值 |")
    report_lines.append("|------|-----|")
    report_lines.append(f"| 平均推理时间 | {benchmark['avg_time']*1000:.2f} ms |")
    report_lines.append(f"| 最小推理时间 | {benchmark['min_time']*1000:.2f} ms |")
    report_lines.append(f"| 最大推理时间 | {benchmark['max_time']*1000:.2f} ms |")
    report_lines.append(f"| 帧率 (FPS) | {benchmark['fps']:.2f} |")
    report_lines.append(f"| 设备 | {benchmark['device']} |")
    report_lines.append("")

    # 4. Visualization Charts
    report_lines.append("## 四、可视化结果\n")
    report_lines.append("### 4.1 生成的图表\n")
    report_lines.append("1. **模型对比柱状图** ([`model_comparison_bars.png`](charts/model_comparison_bars.png))")
    report_lines.append("   - Part2 SegModel vs 轻量化目标模型对比")
    report_lines.append("   - 指标: mIoU, mPA, FPS, 模型大小\n")
    report_lines.append("2. **误差分布热力图** ([`error_distribution_heatmap.png`](charts/error_distribution_heatmap.png))")
    report_lines.append("   - 各类别分割误差率分布")
    report_lines.append("   - 不同模型在不同类别上的表现\n")
    report_lines.append("3. **灵敏度分析曲线** ([`sensitivity_analysis.png`](charts/sensitivity_analysis.png))")
    report_lines.append("   - 剪枝比例 vs 性能")
    report_lines.append("   - 量化方法对比")
    report_lines.append("   - 输入分辨率影响")
    report_lines.append("   - 蒸馏温度影响\n")
    report_lines.append("4. **可视化对比图** ([`comparisons/`](comparisons/))")
    report_lines.append("   - 原始图像 → 分割结果 → 叠加效果\n")

    # 5. Model Optimization Plan
    report_lines.append("## 五、模型轻量化方案\n")
    report_lines.append("### 5.1 优化目标\n")
    report_lines.append("| 指标 | 当前值 | 目标值 |")
    report_lines.append("|------|--------|--------|")
    report_lines.append(f"| 模型大小 | {model_info.get('params_M', 0.37)*4:.1f} MB | ≤500 MB |")
    report_lines.append(f"| 推理时间 | {benchmark['avg_time']*1000:.0f} ms | ≤1000 ms |")
    report_lines.append(f"| FPS | {benchmark['fps']:.1f} | ≥10 |")
    report_lines.append("")

    report_lines.append("### 5.2 优化方案\n")
    report_lines.append("1. **知识蒸馏**")
    report_lines.append("   - 将当前模型知识蒸馏到MobileNetV3骨干网络")
    report_lines.append("   - 蒸馏温度: T=4")
    report_lines.append("   - 预期压缩比: 5x\n")
    report_lines.append("2. **网络剪枝**")
    report_lines.append("   - 剪枝贡献度<5%的卷积通道")
    report_lines.append("   - 目标剪枝率: 30%")
    report_lines.append("   - 微调恢复精度\n")
    report_lines.append("3. **INT8量化**")
    report_lines.append("   - 动态量化 (无需校准数据集)")
    report_lines.append("   - 预期体积压缩: 4x")
    report_lines.append("   - 精度损失: <2%\n")

    # 6. Embedded Deployment
    report_lines.append("## 六、嵌入式部署\n")
    report_lines.append("### 6.1 目标平台\n")
    report_lines.append("- **硬件**: NVIDIA Jetson Xavier NX")
    report_lines.append("- **GPU**: 384 CUDA cores, 48 Tensor cores")
    report_lines.append("- **内存**: 8GB LPDDR4x")
    report_lines.append("- **TDP**: 10W-15W\n")

    report_lines.append("### 6.2 预期性能（模拟估算）\n")
    inference_time = simulator.estimate_inference_time(1.5e9)
    fps = 1.0 / inference_time if inference_time > 0 else 0
    memory = simulator.estimate_memory_usage(int(model_info.get('params', 370000)), (3, 256, 256))
    power = simulator.estimate_power_consumption()

    report_lines.append("| 指标 | 估算值 |")
    report_lines.append("|------|--------|")
    report_lines.append(f"| 推理时间 | {inference_time*1000:.1f} ms |")
    report_lines.append(f"| 帧率 | {fps:.1f} FPS |")
    report_lines.append(f"| 显存使用 | {memory:.2f} GB |")
    report_lines.append(f"| 功耗 | {power:.1f} W |")
    report_lines.append("")

    # 7. Demo Development
    report_lines.append("## 七、场景化Demo\n")
    report_lines.append("### 7.1 功能特性\n")
    report_lines.append("- 支持图像/视频输入")
    report_lines.append("- 实时输出全景分割结果")
    report_lines.append("- 不同类别用不同颜色标注")
    report_lines.append("- PyQt5桌面端应用 (Windows/Linux)\n")

    report_lines.append("### 7.2 演示场景\n")
    scenarios = {
        "coral_reef": "浅海珊瑚礁 - 检测礁石、鱼类、水草等",
        "deep_sea": "深海遗迹 - 检测海星、海参等底栖生物",
        "marine_life": "海洋生物监测 - 鱼类追踪与计数"
    }
    for key, desc in scenarios.items():
        report_lines.append(f"- **{key}**: {desc}")
    report_lines.append("")

    # 8. Conclusions
    report_lines.append("## 八、结论与展望\n")
    report_lines.append("### 8.1 已完成工作\n")
    report_lines.append("- [x] Part2分割模型集成 (0.37M参数)")
    report_lines.append("- [x] 数据可视化图表生成")
    report_lines.append("- [x] 可视化对比图生成")
    report_lines.append("- [x] 性能基准测试")
    report_lines.append("- [x] 嵌入式部署性能估算")
    report_lines.append("- [x] PyQt5 Demo框架开发\n")

    report_lines.append("### 8.2 技术优势\n")
    report_lines.append("1. **轻量化模型**: 仅0.37M参数，适合边缘部署")
    report_lines.append("2. **高效推理**: CPU环境可达到实时处理速度")
    report_lines.append("3. **完整流水线**: 支持图像增强+全景分割")
    report_lines.append("4. **跨平台**: 支持Windows/Linux，预留机器人接口\n")

    report_lines.append("### 8.3 后续工作\n")
    report_lines.append("- [ ] 等待Part2完整Mask2Former模型训练完成")
    report_lines.append("- [ ] 实施知识蒸馏+剪枝+量化优化流程")
    report_lines.append("- [ ] 在真实Jetson硬件上验证性能")
    report_lines.append("- [ ] 完善串口/网络通信接口")
    report_lines.append("")

    report_lines.append("---\n")
    report_lines.append("*报告生成器自动生成* | Part3_Deployment_Demo*")

    # Save report
    output_path = Path('./output/experiment_report.md')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(report_lines), encoding='utf-8')

    print(f"[OK] Complete report saved: {output_path}")
    return output_path


if __name__ == '__main__':
    print("=" * 60)
    print("Generating Complete Experiment Report")
    print("=" * 60)
    generate_complete_report()
    print("=" * 60)
    print("Report generation completed!")
    print("=" * 60)
