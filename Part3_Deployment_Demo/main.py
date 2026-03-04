"""
Demo主程序 - 水下图像增强+全景分割系统
"""

import sys
import os
from pathlib import Path

# 添加模块路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / '05_Shared'))
sys.path.insert(0, str(project_root / '04_Demo' / 'src'))
sys.path.insert(0, str(project_root / '03_EmbeddedDeployment' / 'src'))
sys.path.insert(0, str(project_root / '01_DataVisualization' / 'src'))

import numpy as np
from PIL import Image


def run_gui_demo():
    """运行PyQt5 GUI Demo"""
    print("启动PyQt5 Demo...")
    try:
        from PyQt5.QtWidgets import QApplication
        from main_window import MainWindow
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except ImportError:
        print("错误: PyQt5未安装，请运行: pip install PyQt5")
        sys.exit(1)


def run_benchmark():
    """运行性能测试"""
    print("性能测试...")
    from inference_engine import InferenceEngine
    
    engine = InferenceEngine(device='cpu')
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    benchmark = engine.benchmark(test_image, num_runs=50)
    
    print("\n性能测试结果:")
    print(f"  平均时间: {benchmark['avg_time']*1000:.2f} ms")
    print(f"  最小时间: {benchmark['min_time']*1000:.2f} ms")
    print(f"  最大时间: {benchmark['max_time']*1000:.2f} ms")
    print(f"  标准差: {benchmark['std_time']*1000:.2f} ms")
    print(f"  帧率: {benchmark['fps']:.2f} FPS")
    print(f"  设备: {benchmark['device']}")
    
    info = engine.get_model_info()
    print("\n模型信息:")
    for k, v in info['segmentor'].items():
        print(f"  {k}: {v}")


def run_deployment_simulation():
    """运行嵌入式部署模拟"""
    print("嵌入式部署模拟...")
    from inference_engine import InferenceEngine
    from deployment_report import generate_deployment_report
    
    engine = InferenceEngine(device='cpu')
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    benchmark = engine.benchmark(test_image, num_runs=10)
    
    report_path = generate_deployment_report(
        engine.get_model_info()['segmentor'],
        benchmark,
        "./output/embedded_deployment_report.txt"
    )
    
    print(f"\n报告已生成: {report_path}")
    
    # 显示报告
    report = report_path.read_text(encoding='utf-8')
    print("\n" + "="*60)
    lines = report.split('\n')
    for line in lines[:80]:
        print(line)


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='水下图像增强+全景分割系统')
    parser.add_argument('mode', nargs='?', default='gui',
                       choices=['gui', 'benchmark', 'simulate'],
                       help='运行模式')
    args = parser.parse_args()

    os.makedirs('./output', exist_ok=True)

    if args.mode == 'gui':
        run_gui_demo()
    elif args.mode == 'benchmark':
        run_benchmark()
    elif args.mode == 'simulate':
        run_deployment_simulation()


if __name__ == '__main__':
    main()
