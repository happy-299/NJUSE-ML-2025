"""
可视化Level 2实验结果
生成消融实验、跨项目实验和假设检验的图表
"""

import os
import json
import argparse
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_ablation_results(ablation_dir: str, save_path: str):
    """
    可视化消融实验结果

    Args:
        ablation_dir: 消融实验结果目录
        save_path: 保存路径
    """
    # 读取汇总结果
    summary_file = os.path.join(ablation_dir, "ablation_summary.json")
    if not os.path.exists(summary_file):
        print(f"未找到 {summary_file}")
        return

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 提取数据
    ablation_types = []
    r2_scores = []
    f1_scores = []
    descriptions = []

    for ablation_type, result in summary.items():
        ablation_types.append(ablation_type)
        descriptions.append(result.get("description", ablation_type))

        test_results = result.get("results", {})
        r2_scores.append(test_results.get("test_reg", {}).get("R2", 0))
        f1_scores.append(test_results.get("test_cls", {}).get("F1", 0))

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # R2 comparison
    bars1 = ax1.barh(descriptions, r2_scores, color="steelblue")
    ax1.set_xlabel("R² Score", fontsize=12)
    ax1.set_title(
        "Ablation Study - Regression Performance (R²)", fontsize=14, fontweight="bold"
    )
    ax1.axvline(x=max(r2_scores), color="red", linestyle="--", alpha=0.5, label="Best")

    # 添加数值标签
    for bar, score in zip(bars1, r2_scores):
        ax1.text(
            score + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=10,
        )

    # F1 comparison
    bars2 = ax2.barh(descriptions, f1_scores, color="coral")
    ax2.set_xlabel("F1 Score", fontsize=12)
    ax2.set_title(
        "Ablation Study - Classification Performance (F1)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.axvline(x=max(f1_scores), color="red", linestyle="--", alpha=0.5, label="Best")

    for bar, score in zip(bars2, f1_scores):
        ax2.text(
            score + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"消融实验图表已保存到: {save_path}")


def plot_cross_project_results(cross_project_dir: str, save_path: str):
    """
    可视化跨项目实验结果

    Args:
        cross_project_dir: 跨项目实验结果目录
        save_path: 保存路径
    """
    # 读取汇总结果
    summary_file = os.path.join(cross_project_dir, "cross_project_summary.json")
    if not os.path.exists(summary_file):
        print(f"未找到 {summary_file}")
        return

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 提取数据
    results = summary.get("results", [])
    test_projects = [r["test_project"] for r in results]

    metrics = {
        "R²": [r["test_reg"]["R2"] for r in results],
        "RMSE": [r["test_reg"]["RMSE"] for r in results],
        "MAE": [r["test_reg"]["MAE"] for r in results],
        "Accuracy": [r["test_cls"]["Accuracy"] for r in results],
        "F1": [r["test_cls"]["F1"] for r in results],
    }

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = plt.cm.Set3(np.linspace(0, 1, len(test_projects)))

    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        bars = ax.bar(
            test_projects, values, color=colors, edgecolor="black", linewidth=1.5
        )

        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(
            y=mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.3f}",
        )

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(
            f"Cross-Project Performance - {metric_name}", fontsize=13, fontweight="bold"
        )
        ax.legend(loc="best")
        ax.tick_params(axis="x", rotation=45)

    # 隐藏多余的子图
    axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"跨项目实验图表已保存到: {save_path}")


def plot_hypothesis_test_summary(hypothesis_dir: str, save_path: str):
    """
    可视化假设检验结果汇总

    Args:
        hypothesis_dir: 假设检验结果目录
        save_path: 保存路径
    """
    # 查找所有假设检验结果文件
    result_files = [
        f
        for f in os.listdir(hypothesis_dir)
        if f.startswith("hypothesis_test_") and f.endswith(".json")
    ]

    if not result_files:
        print(f"未找到假设检验结果文件")
        return

    # 创建汇总表格
    summary_data = []

    for result_file in result_files:
        with open(
            os.path.join(hypothesis_dir, result_file), "r", encoding="utf-8"
        ) as f:
            result = json.load(f)

        metric = result.get("metric", "Unknown")
        friedman = result.get("friedman_test", {})

        summary_data.append(
            {
                "指标": metric,
                "Friedman统计量": friedman.get("statistic", 0),
                "p值": friedman.get("p_value", 1),
                "显著性": "是" if friedman.get("significant", False) else "否",
            }
        )

    df = pd.DataFrame(summary_data)

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.2, 0.3, 0.2, 0.2],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # 设置行颜色
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#E7E6E6")

            # 高亮显著的结果
            if j == 3 and df.iloc[i - 1, j] == "是":
                table[(i, j)].set_facecolor("#FFC000")
                table[(i, j)].set_text_props(weight="bold")

    plt.title("Hypothesis Test Results Summary", fontsize=16, fontweight="bold", pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Hypothesis test summary chart saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="可视化Level 2实验结果")
    parser.add_argument(
        "--ablation_dir",
        type=str,
        default="outputs/ablation_mmoe",
        help="消融实验结果目录",
    )
    parser.add_argument(
        "--cross_project_dir",
        type=str,
        default="outputs/cross_project",
        help="跨项目实验结果目录",
    )
    parser.add_argument(
        "--hypothesis_dir",
        type=str,
        default="outputs/hypothesis_test",
        help="假设检验结果目录",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/figures_level2", help="输出图表目录"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("生成Level 2实验可视化")
    print("=" * 60)

    # 消融实验
    if os.path.exists(args.ablation_dir):
        print("\n生成消融实验图表...")
        plot_ablation_results(
            args.ablation_dir, os.path.join(args.output_dir, "ablation_comparison.png")
        )
    else:
        print(f"\n跳过消融实验（目录不存在: {args.ablation_dir}）")

    # 跨项目实验
    if os.path.exists(args.cross_project_dir):
        print("\n生成跨项目实验图表...")
        plot_cross_project_results(
            args.cross_project_dir,
            os.path.join(args.output_dir, "cross_project_comparison.png"),
        )
    else:
        print(f"\n跳过跨项目实验（目录不存在: {args.cross_project_dir}）")

    # 假设检验
    if os.path.exists(args.hypothesis_dir):
        print("\n生成假设检验汇总图表...")
        plot_hypothesis_test_summary(
            args.hypothesis_dir,
            os.path.join(args.output_dir, "hypothesis_test_summary.png"),
        )
    else:
        print(f"\n跳过假设检验（目录不存在: {args.hypothesis_dir}）")

    print("\n" + "=" * 60)
    print("可视化完成！")
    print(f"图表已保存到: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
