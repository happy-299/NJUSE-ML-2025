"""
假设检验实验
使用Friedman test和Nemenyi post-hoc test比较多个模型的性能
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_experiment_results(
    result_dirs: List[str], metric: str = "R2"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    加载实验结果

    Args:
        result_dirs: 结果目录列表
        metric: 要比较的指标（R2, MAE, RMSE, Accuracy, F1等）

    Returns:
        (数据框, 模型名称列表)
    """
    # 首先检查是否为跨项目实验结果目录（包含train_*.json文件）
    is_cross_project = []
    for result_dir in result_dirs:
        cross_project_files = [
            f
            for f in os.listdir(result_dir)
            if f.endswith(".json") and f.startswith("train_")
        ]
        is_cross_project.append(len(cross_project_files) > 0)

    # 如果所有目录都是跨项目实验结果，使用跨项目加载方式
    if all(is_cross_project):
        print("检测到跨项目实验结果，使用跨项目加载方式")
        return load_multiple_cross_project_results(result_dirs, metric)

    # 否则使用原有的单项目加载方式
    data = []
    model_names = []

    for result_dir in result_dirs:
        # 获取模型名称
        model_name = os.path.basename(result_dir)
        model_names.append(model_name)

        # 加载结果文件
        results_file = os.path.join(result_dir, "results.json")
        if not os.path.exists(results_file):
            print(f"警告: {results_file} 不存在，跳过")
            continue

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        # 提取指标值
        if metric in ["R2", "MAE", "MSE", "RMSE"]:
            value = results.get("test_reg", {}).get(metric)
        elif metric in ["Accuracy", "Precision", "Recall", "F1", "F1_macro"]:
            value = results.get("test_cls", {}).get(metric)
        else:
            raise ValueError(f"未知的指标: {metric}")

        if value is not None:
            data.append(value)
        else:
            print(f"警告: 在 {result_dir} 中找不到指标 {metric}")

    # 创建数据框（每列是一个模型）
    df = pd.DataFrame([data], columns=model_names)
    return df, model_names


def load_multiple_cross_project_results(
    result_dirs: List[str], metric: str = "R2"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    加载多个跨项目实验结果（用于模型间比较）

    Args:
        result_dirs: 跨项目实验结果目录列表（每个目录对应一个模型）
        metric: 要比较的指标

    Returns:
        (数据框, 模型名称列表) - 每行是一个测试项目，每列是一个模型
    """
    model_names = []
    all_data = {}  # {test_project: {model_name: value}}

    for result_dir in result_dirs:
        model_name = os.path.basename(result_dir)
        model_names.append(model_name)

        # 查找所有结果文件
        result_files = [
            f
            for f in os.listdir(result_dir)
            if f.endswith(".json") and f.startswith("train_")
        ]

        if not result_files:
            print(f"警告: 在 {result_dir} 中未找到跨项目实验结果文件")
            continue

        # 读取每个测试项目的结果
        for result_file in result_files:
            with open(
                os.path.join(result_dir, result_file), "r", encoding="utf-8"
            ) as f:
                result = json.load(f)

            test_project = result["test_project"]

            # 提取指标值
            if metric in ["R2", "MAE", "MSE", "RMSE"]:
                value = result.get("test_reg", {}).get(metric)
            elif metric in ["Accuracy", "Precision", "Recall", "F1", "F1_macro"]:
                value = result.get("test_cls", {}).get(metric)
            else:
                raise ValueError(f"未知的指标: {metric}")

            if test_project not in all_data:
                all_data[test_project] = {}
            all_data[test_project][model_name] = value

    # 转换为DataFrame（行=测试项目，列=模型）
    test_projects = sorted(all_data.keys())
    data_matrix = []
    for test_project in test_projects:
        row = [all_data[test_project].get(model_name) for model_name in model_names]
        data_matrix.append(row)

    df = pd.DataFrame(data_matrix, columns=model_names, index=test_projects)

    print(f"\n加载了 {len(test_projects)} 个测试项目的结果:")
    print(f"  测试项目: {', '.join(test_projects)}")
    print(f"  模型: {', '.join(model_names)}")

    return df, model_names


def load_cross_project_results(
    result_dir: str, metric: str = "R2"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    加载跨项目实验结果

    Args:
        result_dir: 跨项目实验结果目录
        metric: 要比较的指标

    Returns:
        (数据框, 模型名称列表) - 每行是一个测试项目，每列是一个模型
    """
    # 查找所有结果文件
    result_files = [
        f
        for f in os.listdir(result_dir)
        if f.endswith(".json") and f.startswith("train_")
    ]

    if not result_files:
        raise ValueError(f"在 {result_dir} 中未找到结果文件")

    # 按测试项目组织数据
    data_by_test_project = {}

    for result_file in result_files:
        with open(os.path.join(result_dir, result_file), "r", encoding="utf-8") as f:
            result = json.load(f)

        test_project = result["test_project"]

        # 提取指标值
        if metric in ["R2", "MAE", "MSE", "RMSE"]:
            value = result.get("test_reg", {}).get(metric)
        elif metric in ["Accuracy", "Precision", "Recall", "F1", "F1_macro"]:
            value = result.get("test_cls", {}).get(metric)
        else:
            raise ValueError(f"未知的指标: {metric}")

        if test_project not in data_by_test_project:
            data_by_test_project[test_project] = []

        data_by_test_project[test_project].append(value)

    # 创建数据框
    df = pd.DataFrame(data_by_test_project).T
    df.columns = [f"Model_{i+1}" for i in range(len(df.columns))]

    return df, list(df.columns)


def friedman_test(data: pd.DataFrame) -> Dict[str, Any]:
    """
    执行Friedman检验

    Args:
        data: 数据框，行为数据集，列为算法

    Returns:
        检验结果字典
    """
    # 执行Friedman检验
    statistic, p_value = friedmanchisquare(*[data[col].values for col in data.columns])

    # 计算平均秩
    ranks = data.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean(axis=0).to_dict()

    result = {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "mean_ranks": mean_ranks,
        "interpretation": "存在显著差异" if p_value < 0.05 else "无显著差异",
    }

    return result


def nemenyi_test(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    执行Nemenyi post-hoc检验

    Args:
        data: 数据框，行为数据集，列为算法

    Returns:
        (p值矩阵, 结果字典)
    """
    # 执行Nemenyi检验
    p_values = sp.posthoc_nemenyi_friedman(data)

    # 找出显著差异的模型对
    significant_pairs = []
    for i in range(len(p_values)):
        for j in range(i + 1, len(p_values)):
            if p_values.iloc[i, j] < 0.05:
                model_i = p_values.index[i]
                model_j = p_values.columns[j]
                significant_pairs.append(
                    {
                        "model_1": model_i,
                        "model_2": model_j,
                        "p_value": float(p_values.iloc[i, j]),
                    }
                )

    result = {
        "p_values": p_values.to_dict(),
        "significant_pairs": significant_pairs,
        "num_significant_pairs": len(significant_pairs),
    }

    return p_values, result


def visualize_two_model_comparison(
    x: np.ndarray,
    y: np.ndarray,
    model_names: List[str],
    metric: str,
    comparison_results: Dict[str, Any],
    save_path: str,
):
    """
    可视化两个模型的比较结果

    Args:
        x: 模型1的数据
        y: 模型2的数据
        model_names: 模型名称列表
        metric: 指标名称
        comparison_results: 比较结果字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 箱线图比较
    ax = axes[0, 0]
    bp = ax.boxplot([x, y], labels=model_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["lightblue", "lightcoral"]):
        patch.set_facecolor(color)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Distribution Comparison")
    ax.grid(True, alpha=0.3)

    # 2. 散点图（配对比较）
    ax = axes[0, 1]
    ax.scatter(x, y, alpha=0.6, s=100)
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="y=x")
    ax.set_xlabel(model_names[0])
    ax.set_ylabel(model_names[1])
    ax.set_title("Paired Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 差异分布
    ax = axes[1, 0]
    diff = x - y
    ax.hist(diff, bins=10, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No difference")
    ax.axvline(
        diff.mean(),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Mean diff: {diff.mean():.4f}",
    )
    ax.set_xlabel(f"{model_names[0]} - {model_names[1]}")
    ax.set_ylabel("Frequency")
    ax.set_title("Difference Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 统计结果文本
    ax = axes[1, 1]
    ax.axis("off")

    text_lines = [
        f"Statistical Comparison Results",
        f"{'='*40}",
        f"Metric: {metric}",
        f"",
        f"Model 1: {model_names[0]}",
        f"  Mean ± Std: {comparison_results['descriptive_stats']['model_1_mean']:.4f} ± {comparison_results['descriptive_stats']['model_1_std']:.4f}",
        f"",
        f"Model 2: {model_names[1]}",
        f"  Mean ± Std: {comparison_results['descriptive_stats']['model_2_mean']:.4f} ± {comparison_results['descriptive_stats']['model_2_std']:.4f}",
        f"",
        f"Difference: {comparison_results['descriptive_stats']['mean_difference']:.4f}",
        f"",
        f"Effect Size (Cohen's d): {comparison_results['effect_size']['cohens_d']:.4f}",
        f"  Interpretation: {comparison_results['effect_size']['interpretation']}",
        f"",
    ]

    # 添加检验结果
    if (
        "wilcoxon" in comparison_results
        and "error" not in comparison_results["wilcoxon"]
    ):
        wilcoxon = comparison_results["wilcoxon"]
        text_lines.extend(
            [
                f"Wilcoxon Test:",
                f"  p-value: {wilcoxon['p_value']:.4f}",
                f"  Result: {wilcoxon['interpretation']}",
                f"",
            ]
        )

    if (
        "paired_ttest" in comparison_results
        and "error" not in comparison_results["paired_ttest"]
    ):
        ttest = comparison_results["paired_ttest"]
        text_lines.extend(
            [
                f"Paired t-test:",
                f"  p-value: {ttest['p_value']:.4f}",
                f"  Result: {ttest['interpretation']}",
            ]
        )

    text = "\n".join(text_lines)
    ax.text(
        0.1,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"两模型比较可视化已保存到: {save_path}")


def visualize_nemenyi(p_values: pd.DataFrame, save_path: str):
    """
    可视化Nemenyi检验结果

    Args:
        p_values: p值矩阵
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))

    # 创建热力图
    mask = np.triu(np.ones_like(p_values, dtype=bool))
    sns.heatmap(
        p_values,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        square=True,
        cbar_kws={"label": "p-value"},
    )

    plt.title("Nemenyi Post-hoc Test Results\n(darker = more significant)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Nemenyi检验可视化已保存到: {save_path}")


def run_hypothesis_test(
    result_dirs: List[str],
    output_dir: str,
    metric: str = "R2",
    use_cross_project: bool = False,
):
    """
    运行假设检验

    Args:
        result_dirs: 结果目录列表（如果use_cross_project=True，则只需一个目录）
        output_dir: 输出目录
        metric: 要比较的指标
        use_cross_project: 是否使用跨项目实验结果
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    if use_cross_project:
        if len(result_dirs) != 1:
            raise ValueError("跨项目模式下只需提供一个结果目录")
        data, model_names = load_cross_project_results(result_dirs[0], metric)
        print(f"\n从跨项目实验加载数据:")
        print(f"  测试项目数: {len(data)}")
        print(f"  模型数: {len(model_names)}")
    else:
        data, model_names = load_experiment_results(result_dirs, metric)
        # 自动检测是否为跨项目实验结果
        if len(data) > 1:  # 多行表示多个测试项目
            print(f"\n从跨项目实验加载数据（多模型比较）:")
            print(f"  测试项目数: {len(data)}")
            print(f"  模型数: {len(model_names)}")
        else:
            print(f"\n从单项目实验加载数据:")
            print(f"  模型数: {len(model_names)}")

    print(f"\n数据预览:")
    print(data)

    # 执行Friedman检验
    print(f"\n{'='*60}")
    print("执行 Friedman 检验")
    print(f"{'='*60}")

    # 根据模型数量选择合适的统计检验方法
    n_models = data.shape[1]
    model_names = list(data.columns)

    if n_models < 2:
        # 只有一个模型的结果，输出描述性统计并保存
        print("只有一个模型的结果，无法做 Friedman 或配对检验。保存描述性统计。")
        desc = data.describe().to_dict()
        results = {"metric": metric, "n_models": n_models, "describe": desc}
        out_path = os.path.join(output_dir, f"hypothesis_test_{metric}_desc.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"描述性统计已保存到: {out_path}")
        return results

    if n_models == 2:
        # 使用多种配对检验方法进行两模型比较
        print("检测到两个模型，使用配对检验方法")
        from scipy.stats import wilcoxon, ttest_rel
        import numpy as np

        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values

        # 检查样本数量
        n_samples = len(x)
        if n_samples < 2:
            print(f"\n警告: 样本数量不足 (n={n_samples})，无法进行配对检验。")
            print("提示: 每个模型至少需要2个独立的测量值才能计算标准差和进行统计检验。")
            print("      当前每个模型只有1个结果值（可能来自单次训练）。")
            print("\n建议:")
            print("  1. 使用不同随机种子运行多次训练（例如5-10次）")
            print("  2. 或使用交叉验证获取多个fold的结果")
            print("  3. 或进行跨项目实验（每个项目作为一个样本）")

            # 保存简单对比结果
            comparison_results = {
                "model_1": model_names[0],
                "model_2": model_names[1],
                "n_samples": n_samples,
                "warning": "样本数量不足，无法进行统计检验",
                "single_point_comparison": {
                    "model_1_value": float(x[0]),
                    "model_2_value": float(y[0]),
                    "difference": float(x[0] - y[0]),
                    "better_model": model_names[0] if x[0] > y[0] else model_names[1],
                },
            }

            out_path = os.path.join(output_dir, f"comparison_{metric}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(comparison_results, f, ensure_ascii=False, indent=2)
            print(f"\n单点对比结果已保存到: {out_path}")
            return comparison_results

        # 计算差异
        diff = x - y
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)

        # 计算效应量 (Cohen's d)
        if std_diff > 0 and not np.isnan(std_diff):
            cohens_d = mean_diff / std_diff
        else:
            cohens_d = 0.0

        # 处理 NaN 值
        def safe_float(val):
            if np.isnan(val) or np.isinf(val):
                return None
            return float(val)

        comparison_results = {
            "model_1": model_names[0],
            "model_2": model_names[1],
            "n_samples": int(n_samples),
            "descriptive_stats": {
                "model_1_mean": safe_float(np.mean(x)),
                "model_1_std": safe_float(np.std(x, ddof=1)),
                "model_2_mean": safe_float(np.mean(y)),
                "model_2_std": safe_float(np.std(y, ddof=1)),
                "mean_difference": safe_float(mean_diff),
                "std_difference": safe_float(std_diff),
            },
            "effect_size": {
                "cohens_d": safe_float(cohens_d),
                "interpretation": (
                    (
                        "negligible"
                        if abs(cohens_d) < 0.2
                        else (
                            "small"
                            if abs(cohens_d) < 0.5
                            else "medium" if abs(cohens_d) < 0.8 else "large"
                        )
                    )
                    if not np.isnan(cohens_d)
                    else "undefined"
                ),
            },
        }

        # Wilcoxon signed-rank test (非参数检验)
        print("\n1. Wilcoxon Signed-Rank Test (非参数检验)")
        try:
            stat, p_value = wilcoxon(x, y, alternative="two-sided")
            comparison_results["wilcoxon"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),  # 转换为 Python bool
                "interpretation": "显著差异" if p_value < 0.05 else "无显著差异",
            }
            print(f"   统计量: {stat:.4f}")
            print(f"   p值: {p_value:.4f}")
            print(f"   结论: {'显著差异' if p_value < 0.05 else '无显著差异'} (α=0.05)")
        except ValueError as e:
            comparison_results["wilcoxon"] = {"error": str(e)}
            print(f"   错误: {e}")

        # Paired t-test (参数检验，假设正态分布)
        print("\n2. Paired t-test (参数检验)")
        try:
            stat, p_value = ttest_rel(x, y)
            # 处理 NaN 值
            if np.isnan(stat) or np.isnan(p_value):
                comparison_results["paired_ttest"] = {
                    "error": "无法计算（标准差为0或样本数不足）"
                }
                print(f"   错误: 无法计算（标准差为0或样本数不足）")
            else:
                comparison_results["paired_ttest"] = {
                    "statistic": float(stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05),  # 转换为 Python bool
                    "interpretation": "显著差异" if p_value < 0.05 else "无显著差异",
                }
                print(f"   统计量: {stat:.4f}")
                print(f"   p值: {p_value:.4f}")
                print(
                    f"   结论: {'显著差异' if p_value < 0.05 else '无显著差异'} (α=0.05)"
                )
        except Exception as e:
            comparison_results["paired_ttest"] = {"error": str(e)}
            print(f"   错误: {e}")

        # 打印描述性统计和效应量
        print("\n3. 描述性统计")
        stats = comparison_results["descriptive_stats"]
        if stats["model_1_mean"] is not None:
            print(
                f"   {model_names[0]}: {stats['model_1_mean']:.4f} ± {stats['model_1_std'] if stats['model_1_std'] is not None else 'N/A'}"
            )
        else:
            print(f"   {model_names[0]}: N/A")

        if stats["model_2_mean"] is not None:
            print(
                f"   {model_names[1]}: {stats['model_2_mean']:.4f} ± {stats['model_2_std'] if stats['model_2_std'] is not None else 'N/A'}"
            )
        else:
            print(f"   {model_names[1]}: N/A")

        if stats["mean_difference"] is not None:
            print(
                f"   平均差异: {stats['mean_difference']:.4f} ± {stats['std_difference'] if stats['std_difference'] is not None else 'N/A'}"
            )
        else:
            print(f"   平均差异: N/A")

        print("\n4. 效应量")
        if comparison_results["effect_size"]["cohens_d"] is not None:
            print(f"   Cohen's d: {comparison_results['effect_size']['cohens_d']:.4f}")
        else:
            print(f"   Cohen's d: N/A")
        print(f"   解释: {comparison_results['effect_size']['interpretation']}")

        # 保存结果
        out_path = os.path.join(output_dir, f"comparison_{metric}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "comparison_results": comparison_results,
                    "raw_data": {
                        model_names[0]: x.tolist(),
                        model_names[1]: y.tolist(),
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\n两模型比较结果已保存到: {out_path}")

        # 生成可视化
        viz_path = os.path.join(output_dir, f"comparison_{metric}.png")
        visualize_two_model_comparison(
            x, y, model_names, metric, comparison_results, viz_path
        )

        return comparison_results

    # n_models >= 3，使用 Friedman + Nemenyi（原有逻辑）
    friedman_result = friedman_test(data)

    print(f"\nFriedman 检验结果:")
    print(f"  统计量: {friedman_result['statistic']:.4f}")
    print(f"  p值: {friedman_result['p_value']:.4f}")
    print(f"  结论: {friedman_result['interpretation']}")
    print(f"\n平均秩次:")
    for model, rank in sorted(
        friedman_result["mean_ranks"].items(), key=lambda x: x[1]
    ):
        print(f"  {model}: {rank:.2f}")

    # 如果Friedman检验显著，执行Nemenyi post-hoc检验
    if friedman_result["significant"]:
        print(f"\n{'='*60}")
        print("执行 Nemenyi Post-hoc 检验")
        print(f"{'='*60}")

        p_values, nemenyi_result = nemenyi_test(data)

        print(f"\nNemenyi 检验结果:")
        print(f"  发现 {nemenyi_result['num_significant_pairs']} 对显著差异的模型")

        if nemenyi_result["significant_pairs"]:
            print("\n显著差异的模型对 (p < 0.05):")
            for pair in nemenyi_result["significant_pairs"]:
                print(
                    f"  {pair['model_1']} vs {pair['model_2']}: p={pair['p_value']:.4f}"
                )

        # 可视化
        viz_path = os.path.join(output_dir, f"nemenyi_{metric}.png")
        visualize_nemenyi(p_values, viz_path)
    else:
        print("\nFriedman检验未发现显著差异，跳过post-hoc检验")
        nemenyi_result = None

    # 保存结果
    results = {
        "metric": metric,
        "use_cross_project": use_cross_project,
        "model_names": model_names,
        "data": data.to_dict(),
        "friedman_test": friedman_result,
        "nemenyi_test": nemenyi_result,
    }

    output_file = os.path.join(output_dir, f"hypothesis_test_{metric}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"假设检验完成！结果已保存到: {output_dir}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="假设检验实验")
    parser.add_argument(
        "--result_dirs",
        type=str,
        nargs="+",
        required=True,
        help="结果目录列表（或跨项目实验的单个目录）",
    )
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--metric",
        type=str,
        default="R2",
        choices=[
            "R2",
            "MAE",
            "MSE",
            "RMSE",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "F1_macro",
        ],
        help="要比较的指标",
    )
    parser.add_argument(
        "--cross_project", action="store_true", help="使用跨项目实验结果"
    )

    args = parser.parse_args()

    run_hypothesis_test(args.result_dirs, args.output, args.metric, args.cross_project)


if __name__ == "__main__":
    main()
