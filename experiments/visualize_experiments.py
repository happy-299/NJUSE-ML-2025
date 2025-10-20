"""
实验结果可视化 - 综合展示消融、泛化和假设检验结果
"""

import os
import json
import argparse
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def plot_ablation_results(summary_file: str, output_dir: str):
    """可视化消融实验结果"""
    with open(summary_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        print("消融实验结果为空")
        return

    # 提取数据
    variants = [r["name"] for r in results]

    # 回归指标
    reg_metrics = ["R2", "RMSE", "MAE"]
    reg_data = {m: [] for m in reg_metrics}

    for r in results:
        for m in reg_metrics:
            val = r.get("test_reg", {}).get(m, np.nan)
            reg_data[m].append(val)

    # 分类指标
    cls_metrics = ["Accuracy", "F1", "F1_macro"]
    cls_data = {m: [] for m in cls_metrics}

    for r in results:
        for m in cls_metrics:
            val = r.get("test_cls", {}).get(m, np.nan)
            cls_data[m].append(val)

    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("消融实验结果对比", fontsize=16, fontweight="bold")

    # 回归指标
    for idx, metric in enumerate(reg_metrics):
        ax = axes[0, idx]
        values = reg_data[metric]
        bars = ax.bar(
            range(len(variants)),
            values,
            alpha=0.7,
            color=plt.cm.Blues(np.linspace(0.4, 0.8, len(variants))),
        )
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"回归: {metric}")
        ax.grid(True, alpha=0.3, axis="y")

        # 标注数值
        for i, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # 分类指标
    for idx, metric in enumerate(cls_metrics):
        ax = axes[1, idx]
        values = cls_data[metric]
        bars = ax.bar(
            range(len(variants)),
            values,
            alpha=0.7,
            color=plt.cm.Greens(np.linspace(0.4, 0.8, len(variants))),
        )
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"分类: {metric}")
        ax.grid(True, alpha=0.3, axis="y")

        # 标注数值
        for i, (bar, val) in enumerate(zip(bars, values)):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"消融实验对比图已保存: {output_path}")
    plt.close()


def plot_cross_project_results(summary_file: str, output_dir: str):
    """可视化跨项目实验结果"""
    if not os.path.exists(summary_file):
        print(f"跨项目实验结果不存在: {summary_file}")
        return

    with open(summary_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        print("跨项目实验结果为空")
        return

    # 提取数据
    scenarios = []
    reg_r2 = []
    cls_acc = []

    for r in results:
        scenario = r.get("scenario", "unknown")
        project = r.get("project", r.get("test_project", "unknown"))
        label = f"{scenario}\n({project})"
        scenarios.append(label)

        reg_r2.append(r.get("test_reg", {}).get("R2", np.nan))
        cls_acc.append(r.get("test_cls", {}).get("Accuracy", np.nan))

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("泛化性实验结果", fontsize=16, fontweight="bold")

    # R2
    ax = axes[0]
    bars = ax.bar(range(len(scenarios)), reg_r2, alpha=0.7, color="skyblue")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_ylabel("R2 Score")
    ax.set_title("回归任务 - R2")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    for bar, val in zip(bars, reg_r2):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
            )

    # Accuracy
    ax = axes[1]
    bars = ax.bar(range(len(scenarios)), cls_acc, alpha=0.7, color="lightgreen")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("分类任务 - Accuracy")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, cls_acc):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "cross_project_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"跨项目实验对比图已保存: {output_path}")
    plt.close()


def create_comprehensive_table(
    ablation_file: str, cross_project_file: str, output_dir: str
):
    """创建综合结果表格"""
    data = []

    # 消融实验
    if os.path.exists(ablation_file):
        with open(ablation_file, "r", encoding="utf-8") as f:
            ablation_results = json.load(f)

        for r in ablation_results:
            row = {
                "实验类型": "消融实验",
                "变体": r.get("name", r.get("variant", "unknown")),
                "R2": r.get("test_reg", {}).get("R2", np.nan),
                "RMSE": r.get("test_reg", {}).get("RMSE", np.nan),
                "MAE": r.get("test_reg", {}).get("MAE", np.nan),
                "Accuracy": r.get("test_cls", {}).get("Accuracy", np.nan),
                "F1": r.get("test_cls", {}).get("F1", np.nan),
                "F1_macro": r.get("test_cls", {}).get("F1_macro", np.nan),
            }
            data.append(row)

    # 跨项目实验
    if os.path.exists(cross_project_file):
        with open(cross_project_file, "r", encoding="utf-8") as f:
            cross_results = json.load(f)

        for r in cross_results:
            scenario = r.get("scenario", "unknown")
            project = r.get("project", r.get("test_project", "unknown"))
            row = {
                "实验类型": "泛化实验",
                "变体": f"{scenario} ({project})",
                "R2": r.get("test_reg", {}).get("R2", np.nan),
                "RMSE": r.get("test_reg", {}).get("RMSE", np.nan),
                "MAE": r.get("test_reg", {}).get("MAE", np.nan),
                "Accuracy": r.get("test_cls", {}).get("Accuracy", np.nan),
                "F1": r.get("test_cls", {}).get("F1", np.nan),
                "F1_macro": r.get("test_cls", {}).get("F1_macro", np.nan),
            }
            data.append(row)

    if not data:
        print("没有可用的实验结果")
        return

    df = pd.DataFrame(data)

    # 保存为 Markdown
    md_path = os.path.join(output_dir, "comprehensive_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 综合实验结果表\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")

    print(f"综合结果表已保存: {md_path}")

    # 保存为 CSV
    csv_path = os.path.join(output_dir, "comprehensive_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"CSV 结果已保存: {csv_path}")


def visualize_all_experiments(
    ablation_summary: str, cross_project_summary: str, output_dir: str
):
    """可视化所有实验结果"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("开始生成可视化结果...")
    print(f"{'='*60}\n")

    # 消融实验
    if os.path.exists(ablation_summary):
        plot_ablation_results(ablation_summary, output_dir)
    else:
        print(f"消融实验结果不存在: {ablation_summary}")

    # 跨项目实验
    if os.path.exists(cross_project_summary):
        plot_cross_project_results(cross_project_summary, output_dir)
    else:
        print(f"跨项目实验结果不存在: {cross_project_summary}")

    # 综合表格
    create_comprehensive_table(ablation_summary, cross_project_summary, output_dir)

    print(f"\n{'='*60}")
    print("可视化完成！")
    print(f"结果目录: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="可视化实验结果")
    parser.add_argument(
        "--ablation",
        type=str,
        default="./outputs/ablation/ablation_summary.json",
        help="消融实验汇总文件",
    )
    parser.add_argument(
        "--cross_project",
        type=str,
        default="./outputs/cross_project/generalization_summary.json",
        help="跨项目实验汇总文件",
    )
    parser.add_argument(
        "--output", type=str, default="./outputs/visualizations", help="输出目录"
    )

    args = parser.parse_args()

    visualize_all_experiments(args.ablation, args.cross_project, args.output)


if __name__ == "__main__":
    main()
