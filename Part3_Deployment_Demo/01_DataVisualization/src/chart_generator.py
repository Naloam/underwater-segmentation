"""
图表生成模块

生成各种实验结果图表：柱状图、热力图、折线图等。
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 使用seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class ChartGenerator:
    """图表生成器"""

    def __init__(self, output_dir: Path = None):
        """
        初始化图表生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir) if output_dir else Path('./output/charts')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_figure(self, fig, filename: str, dpi: int = 300):
        """保存图表"""
        save_path = self.output_dir / filename
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"图表已保存: {save_path}")
        return save_path

    def generate_bar_chart(
        self,
        data: Dict[str, List[float]],
        title: str = "模型性能对比",
        ylabel: str = "指标值",
        filename: str = "bar_chart.png"
    ) -> Path:
        """
        生成柱状图

        Args:
            data: 数据字典 {模型名: [指标值列表]}
            title: 图表标题
            ylabel: Y轴标签
            filename: 保存文件名

        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(data.keys())
        metrics_data = list(data.values())

        x = np.arange(len(models))
        width = 0.8 / len(metrics_data[0])

        for i, values in enumerate(metrics_data):
            offset = (i - len(values) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=f'Metric {i+1}')

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        return self.save_figure(fig, filename)

    def generate_heatmap(
        self,
        data: np.ndarray,
        x_labels: List[str],
        y_labels: List[str],
        title: str = "热力图",
        filename: str = "heatmap.png"
    ) -> Path:
        """
        生成热力图

        Args:
            data: 数据矩阵
            x_labels: X轴标签
            y_labels: Y轴标签
            title: 图表标题
            filename: 保存文件名

        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto')

        # 设置刻度
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        # 添加数值标注
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                            ha="center", va="center", color="black")

        ax.set_title(title, fontsize=14)
        fig.colorbar(im, ax=ax)

        return self.save_figure(fig, filename)

    def generate_line_chart(
        self,
        data: Dict[str, List[float]],
        x_values: List[float],
        title: str = "折线图",
        xlabel: str = "X轴",
        ylabel: str = "Y轴",
        filename: str = "line_chart.png"
    ) -> Path:
        """
        生成折线图

        Args:
            data: 数据字典 {系列名: [y值列表]}
            x_values: X轴值
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            filename: 保存文件名

        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        for series_name, y_values in data.items():
            ax.plot(x_values, y_values, marker='o', label=series_name)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

        return self.save_figure(fig, filename)

    def generate_sensitivity_curve(
        self,
        diffusion_steps: List[int],
        miou_values: List[float],
        clip_lrs: List[float],
        semantic_offset: List[float],
        filename: str = "sensitivity_curve.png"
    ) -> Path:
        """
        生成灵敏度分析曲线

        Args:
            diffusion_steps: 扩散步数列表
            miou_values: 对应的mIoU值
            clip_lrs: CLIP学习率列表
            semantic_offset: 对应的语义偏移率
            filename: 保存文件名

        Returns:
            保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 扩散步数 vs mIoU
        ax1.plot(diffusion_steps, miou_values, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Diffusion Steps', fontsize=12)
        ax1.set_ylabel('mIoU', fontsize=12)
        ax1.set_title('Diffusion Steps Sensitivity', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # CLIP学习率 vs 语义偏移率
        ax2.plot(clip_lrs, semantic_offset, 's-', color='corange',
                linewidth=2, markersize=8)
        ax2.set_xlabel('CLIP Learning Rate', fontsize=12)
        ax2.set_ylabel('Semantic Offset Rate (%)', fontsize=12)
        ax2.set_title('CLIP LR Sensitivity', fontsize=14)
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return self.save_figure(fig, filename)

    def generate_comparison_table(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "模型对比表",
        filename: str = "comparison_table.png"
    ) -> Path:
        """
        生成对比表格图

        Args:
            data: 数据 {模型名: {指标名: 值}}
            title: 标题
            filename: 保存文件名

        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')

        # 转换为DataFrame
        df = pd.DataFrame(data).T

        # 创建表格
        table = ax.table(
            cellText=df.round(4).values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # 设置表头样式
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title(title, fontsize=14, pad=20)

        return self.save_figure(fig, filename)


# 示例使用
if __name__ == '__main__':
    generator = ChartGenerator()

    # 示例数据
    models_data = {
        'Mask2Former': [0.92, 0.88, 0.95, 0.90],
        'SegFormer': [0.85, 0.82, 0.88, 0.80],
        'Lightweight': [0.80, 0.78, 0.83, 0.75]
    }

    # 生成柱状图
    generator.generate_bar_chart(
        models_data,
        title="模型性能对比",
        ylabel="指标值",
        filename="model_comparison.png"
    )

    # 生成灵敏度曲线
    generator.generate_sensitivity_curve(
        diffusion_steps=[100, 300, 500, 700, 1000],
        miou_values=[0.85, 0.88, 0.91, 0.92, 0.91],
        clip_lrs=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        semantic_offset=[5.2, 3.8, 2.5, 2.8, 4.1],
        filename="sensitivity_analysis.png"
    )
