"""
综合图表生成脚本

根据 Part2_Enhanced 真实评估结果生成可视化图表：
1. 消融实验对比柱状图（mIoU, Accuracy, F1）
2. 各类别 IoU 热力图（4 组消融 × 8 类别）
3. 消融实验增益分析折线图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------- 数据来源 ----------
EVAL_DIR = Path(__file__).parent.parent / "Part2_Enhanced" / "eval_results"
METRICS_PATH = EVAL_DIR / "metrics.json"
ABLATION_PATH = EVAL_DIR / "ablation_results.json"

CLASS_NAMES = [
    "aquatic_plants", "fish", "human_divers", "reefs",
    "robots", "sea-floor", "waterbody_bg", "wrecks",
]
CLASS_LABELS_CN = [
    "水生植物\n(0)", "鱼类\n(1)", "潜水员\n(2)", "礁石\n(3)",
    "机器人\n(4)", "海底\n(5)", "水体背景\n(6)", "沉船\n(7)",
]

EXP_LABELS = {
    "baseline": "Baseline\n(仅CNN)",
    "with_clip": "+CLIP\n语义分支",
    "with_diffusion": "+Diffusion\n扩散分支",
    "full": "Full\n(完整模型)",
}


def load_data():
    with open(METRICS_PATH, "r") as f:
        full_metrics = json.load(f)
    with open(ABLATION_PATH, "r") as f:
        ablation = json.load(f)
    return full_metrics, ablation


# ---------- 图表 1: 消融实验柱状图 ----------
def generate_ablation_bars(ablation: dict, output_dir: Path):
    print("\n[1/3] 生成消融实验对比柱状图...")

    exp_names = list(ablation.keys())
    labels = [EXP_LABELS.get(n, n) for n in exp_names]

    miou_vals = [ablation[n]["miou"] * 100 for n in exp_names]
    acc_vals = [ablation[n]["accuracy"] * 100 for n in exp_names]
    f1_vals = [ablation[n]["f1"] * 100 for n in exp_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("消融实验 — 各指标对比", fontsize=16, fontweight="bold")

    colors = ["#90CAF9", "#64B5F6", "#42A5F5", "#1E88E5"]

    for ax, metric_vals, title in zip(
        axes, [miou_vals, acc_vals, f1_vals], ["mIoU (%)", "Pixel Accuracy (%)", "F1 Score (%)"]
    ):
        bars = ax.bar(labels, metric_vals, color=colors, edgecolor="black", linewidth=1.2)
        for bar, val in zip(bars, metric_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        ax.set_ylabel(title, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(metric_vals) * 1.3 + 1)

    plt.tight_layout()
    path = output_dir / "model_comparison_bars.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path}")


# ---------- 图表 2: 各类别 IoU 热力图 ----------
def generate_class_iou_heatmap(ablation: dict, output_dir: Path):
    print("\n[2/3] 生成各类别 IoU 热力图...")

    exp_names = list(ablation.keys())
    labels = [EXP_LABELS.get(n, n) for n in exp_names]
    n_cls = len(CLASS_NAMES)

    # shape: (num_classes, num_experiments)
    iou_matrix = np.zeros((n_cls, len(exp_names)))
    for j, name in enumerate(exp_names):
        for i in range(n_cls):
            iou_matrix[i, j] = ablation[name].get(f"iou_class_{i}", 0.0) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(iou_matrix, cmap="YlOrRd_r", aspect="auto", vmin=0)

    ax.set_xticks(np.arange(len(exp_names)))
    ax.set_yticks(np.arange(n_cls))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(CLASS_LABELS_CN, fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    for i in range(n_cls):
        for j in range(len(exp_names)):
            val = iou_matrix[i, j]
            color = "white" if val < 5 else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("IoU (%)", rotation=270, labelpad=18, fontsize=12, fontweight="bold")

    ax.set_title("消融实验 — 各类别 IoU 分布", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("实验配置", fontsize=12, fontweight="bold")
    ax.set_ylabel("目标类别", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = output_dir / "error_distribution_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path}")


# ---------- 图表 3: 模块增益分析 ----------
def generate_module_gain_chart(ablation: dict, output_dir: Path):
    print("\n[3/3] 生成模块增益分析图...")

    baseline = ablation["baseline"]
    exp_names = ["baseline", "with_clip", "with_diffusion", "full"]
    labels = [EXP_LABELS.get(n, n) for n in exp_names]

    miou_vals = [ablation[n]["miou"] * 100 for n in exp_names]
    acc_vals = [ablation[n]["accuracy"] * 100 for n in exp_names]

    # 相对于 baseline 的增益
    miou_gain = [v - miou_vals[0] for v in miou_vals]
    acc_gain = [v - acc_vals[0] for v in acc_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("模块增益分析（相对 Baseline）", fontsize=16, fontweight="bold")

    x = np.arange(len(exp_names))
    width = 0.4

    # mIoU 增益
    colors_miou = ["#BDBDBD", "#66BB6A", "#42A5F5", "#EF5350"]
    bars1 = ax1.bar(x, miou_gain, width, color=colors_miou, edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars1, miou_gain):
        ax1.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 0.1,
                 f"+{val:.2f}" if val > 0 else f"{val:.2f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("mIoU 增益 (百分点)", fontsize=12, fontweight="bold")
    ax1.set_title("mIoU 增益", fontsize=13)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(0, color="gray", linewidth=0.8)

    # Accuracy 增益
    bars2 = ax2.bar(x, acc_gain, width, color=colors_miou, edgecolor="black", linewidth=1.2)
    for bar, val in zip(bars2, acc_gain):
        ax2.text(bar.get_x() + bar.get_width() / 2, max(val, 0) + 0.3,
                 f"+{val:.2f}" if val > 0 else f"{val:.2f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Accuracy 增益 (百分点)", fontsize=12, fontweight="bold")
    ax2.set_title("Pixel Accuracy 增益", fontsize=13)
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(0, color="gray", linewidth=0.8)

    plt.tight_layout()
    path = output_dir / "sensitivity_analysis.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path}")


def main():
    print("=" * 60)
    print("Part3 — 基于真实评估数据的图表生成")
    print("=" * 60)

    output_dir = Path("./output/charts")
    output_dir.mkdir(parents=True, exist_ok=True)

    full_metrics, ablation = load_data()

    generate_ablation_bars(ablation, output_dir)
    generate_class_iou_heatmap(ablation, output_dir)
    generate_module_gain_chart(ablation, output_dir)

    print("\n" + "=" * 60)
    print("图表生成完成！")
    print("=" * 60)
    print(f"\n输出目录: {output_dir.resolve()}")
    for png in sorted(output_dir.glob("*.png")):
        print(f"  - {png.name}")


if __name__ == "__main__":
    main()
