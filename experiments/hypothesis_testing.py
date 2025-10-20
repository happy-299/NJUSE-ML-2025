"""
假设检验：评估多种算法的统计显著性差异

基于 Demšar (2006) 的推荐方法：
1. Friedman test - 非参数检验，检测多个算法在多个数据集上是否有显著差异
2. Nemenyi post-hoc test - 如果 Friedman test 显著，进行成对比较
3. Kruskal-Wallis test + Dunn's test - 替代方案
"""

import os
import json
import argparse
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, kruskal
import matplotlib.pyplot as plt
import seaborn as sns

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HypothesisTest:
    """假设检验工具类"""

    @staticmethod
    def friedman_test(data: np.ndarray) -> Tuple[float, float]:
        """
        Friedman test

        Args:
            data: shape (n_datasets, n_algorithms) 的性能矩阵

        Returns:
            statistic, p_value
        """
        return friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])

    @staticmethod
    def nemenyi_test(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Nemenyi post-hoc test

        Args:
            data: shape (n_datasets, n_algorithms) 的性能矩阵
            alpha: 显著性水平

        Returns:
            p_values: shape (n_algorithms, n_algorithms) 的成对 p 值矩阵
        """
        n_datasets, n_algorithms = data.shape

        # 计算平均秩
        ranks = np.zeros_like(data)
        for i in range(n_datasets):
            ranks[i] = stats.rankdata(data[i])  # 对每个数据集排序

        mean_ranks = ranks.mean(axis=0)

        # Nemenyi 临界差值
        # CD = q_alpha * sqrt(k(k+1) / (6N))
        # 其中 k 是算法数，N 是数据集数
        from scipy.stats import studentized_range

        q_alpha = studentized_range.ppf(1 - alpha, n_algorithms, np.inf)
        cd = q_alpha * np.sqrt(n_algorithms * (n_algorithms + 1) / (6 * n_datasets))

        # 计算成对差异
        p_values = np.ones((n_algorithms, n_algorithms))
        for i in range(n_algorithms):
            for j in range(i + 1, n_algorithms):
                diff = abs(mean_ranks[i] - mean_ranks[j])
                # 简化：如果差异 > CD，则认为显著（p < alpha）
                if diff > cd:
                    p_values[i, j] = p_values[j, i] = alpha / 2
                else:
                    p_values[i, j] = p_values[j, i] = 1.0

        return p_values, mean_ranks, cd

    @staticmethod
    def kruskal_wallis_test(data: np.ndarray) -> Tuple[float, float]:
        """
        Kruskal-Wallis H-test

        Args:
            data: shape (n_datasets, n_algorithms) 的性能矩阵

        Returns:
            statistic, p_value
        """
        return kruskal(*[data[:, i] for i in range(data.shape[1])])

    @staticmethod
    def dunn_test(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """
        Dunn's post-hoc test (简化版本)

        Args:
            data: shape (n_datasets, n_algorithms) 的性能矩阵
            alpha: 显著性水平

        Returns:
            p_values: shape (n_algorithms, n_algorithms) 的成对 p 值矩阵
        """
        from scipy.stats import mannwhitneyu

        n_algorithms = data.shape[1]
        p_values = np.ones((n_algorithms, n_algorithms))

        for i in range(n_algorithms):
            for j in range(i + 1, n_algorithms):
                # Mann-Whitney U test (Wilcoxon rank-sum test)
                stat, p = mannwhitneyu(data[:, i], data[:, j], alternative="two-sided")

                # Bonferroni 校正
                p_corrected = min(p * (n_algorithms * (n_algorithms - 1) / 2), 1.0)
                p_values[i, j] = p_values[j, i] = p_corrected

        return p_values


def load_results_from_experiments(
    result_dirs: List[str], metric: str = "R2"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    从实验结果目录中加载性能指标

    Args:
        result_dirs: 实验结果目录列表
        metric: 要提取的指标名称

    Returns:
        performance_df: 性能数据框
        algorithm_names: 算法名称列表
    """
    data = []
    algorithm_names = []

    for result_dir in result_dirs:
        # 读取结果文件
        result_file = os.path.join(result_dir, "results.json")
        if not os.path.exists(result_file):
            print(f"警告: 结果文件不存在 - {result_file}")
            continue

        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        # 提取算法名称
        algo_name = os.path.basename(result_dir)
        algorithm_names.append(algo_name)

        # 提取性能指标
        if "test_reg" in results and metric in results["test_reg"]:
            value = results["test_reg"][metric]
        elif "test_cls" in results and metric in results["test_cls"]:
            value = results["test_cls"][metric]
        else:
            print(f"警告: 指标 {metric} 未找到于 {result_dir}")
            value = np.nan

        data.append(value)

    # 创建数据框（单个数据集的情况）
    df = pd.DataFrame([data], columns=algorithm_names)

    return df, algorithm_names


def load_results_from_summary(
    summary_file: str, metric: str = "R2", scenario_key: str = "test_reg"
) -> Tuple[np.ndarray, List[str]]:
    """
    从汇总文件中加载多个实验的结果

    Args:
        summary_file: 汇总 JSON 文件路径
        metric: 要提取的指标
        scenario_key: 结果字典的键（如 "test_reg" 或 "test_cls"）

    Returns:
        data: shape (n_datasets, n_algorithms) 的性能矩阵
        algorithm_names: 算法名称列表
    """
    with open(summary_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 假设 results 是一个列表，每个元素对应一个实验变体
    algorithm_names = []
    data_dict = {}

    for item in results:
        # 提取算法/变体名称
        if "name" in item:
            algo_name = item["name"]
        elif "variant" in item:
            algo_name = item["variant"]
        else:
            algo_name = str(item.get("scenario", "unknown"))

        # 提取指标值
        if scenario_key in item and metric in item[scenario_key]:
            value = item[scenario_key][metric]
        else:
            continue

        if algo_name not in data_dict:
            data_dict[algo_name] = []
            algorithm_names.append(algo_name)

        data_dict[algo_name].append(value)

    # 转换为矩阵（假设每个算法有相同数量的数据点）
    max_len = max(len(v) for v in data_dict.values())
    data = np.array(
        [
            data_dict[name] + [np.nan] * (max_len - len(data_dict[name]))
            for name in algorithm_names
        ]
    ).T

    return data, algorithm_names


def perform_hypothesis_test(
    data: np.ndarray,
    algorithm_names: List[str],
    alpha: float = 0.05,
    test_type: str = "friedman",
) -> Dict[str, Any]:
    """
    执行假设检验

    Args:
        data: shape (n_datasets, n_algorithms) 的性能矩阵
        algorithm_names: 算法名称列表
        alpha: 显著性水平
        test_type: 检验类型 ("friedman" 或 "kruskal")

    Returns:
        检验结果字典
    """
    n_datasets, n_algorithms = data.shape

    print(f"\n{'='*60}")
    print(f"假设检验: {test_type.upper()}")
    print(f"数据集数: {n_datasets}, 算法数: {n_algorithms}")
    print(f"显著性水平 α = {alpha}")
    print(f"{'='*60}\n")

    results = {
        "test_type": test_type,
        "n_datasets": n_datasets,
        "n_algorithms": n_algorithms,
        "algorithm_names": algorithm_names,
        "alpha": alpha,
    }

    # 主检验
    if test_type == "friedman":
        stat, p = HypothesisTest.friedman_test(data)
        results["friedman_statistic"] = float(stat)
        results["friedman_pvalue"] = float(p)

        print(f"Friedman Test:")
        print(f"  统计量 = {stat:.4f}")
        print(f"  p-value = {p:.6f}")

        if p < alpha:
            print(f"  结论: 拒绝原假设 (p < {alpha})，算法间存在显著差异")
            print(f"\n执行 Nemenyi post-hoc test...")

            # Post-hoc test
            p_values, mean_ranks, cd = HypothesisTest.nemenyi_test(data, alpha)
            results["post_hoc"] = "nemenyi"
            results["mean_ranks"] = mean_ranks.tolist()
            results["critical_difference"] = float(cd)
            results["pairwise_pvalues"] = p_values.tolist()

            print(f"\nNemenyi Test:")
            print(f"  临界差值 CD = {cd:.4f}")
            print(f"  平均秩次:")
            for i, name in enumerate(algorithm_names):
                print(f"    {name}: {mean_ranks[i]:.2f}")

            # 显著性对比矩阵
            print(f"\n  成对比较 (显著差异标记为 *):")
            for i in range(n_algorithms):
                for j in range(i + 1, n_algorithms):
                    sig = "*" if p_values[i, j] < alpha else ""
                    print(
                        f"    {algorithm_names[i]} vs {algorithm_names[j]}: "
                        f"p={p_values[i, j]:.4f} {sig}"
                    )
        else:
            print(f"  结论: 不能拒绝原假设 (p >= {alpha})，算法间无显著差异")
            results["post_hoc"] = None

    elif test_type == "kruskal":
        stat, p = HypothesisTest.kruskal_wallis_test(data)
        results["kruskal_statistic"] = float(stat)
        results["kruskal_pvalue"] = float(p)

        print(f"Kruskal-Wallis Test:")
        print(f"  统计量 = {stat:.4f}")
        print(f"  p-value = {p:.6f}")

        if p < alpha:
            print(f"  结论: 拒绝原假设 (p < {alpha})，算法间存在显著差异")
            print(f"\n执行 Dunn's post-hoc test...")

            # Post-hoc test
            p_values = HypothesisTest.dunn_test(data, alpha)
            results["post_hoc"] = "dunn"
            results["pairwise_pvalues"] = p_values.tolist()

            print(f"\nDunn's Test (Bonferroni 校正):")
            print(f"  成对比较:")
            for i in range(n_algorithms):
                for j in range(i + 1, n_algorithms):
                    sig = "*" if p_values[i, j] < alpha else ""
                    print(
                        f"    {algorithm_names[i]} vs {algorithm_names[j]}: "
                        f"p={p_values[i, j]:.4f} {sig}"
                    )
        else:
            print(f"  结论: 不能拒绝原假设 (p >= {alpha})，算法间无显著差异")
            results["post_hoc"] = None

    return results


def visualize_results(
    data: np.ndarray,
    algorithm_names: List[str],
    output_path: str,
    test_results: Dict[str, Any] = None,
):
    """可视化假设检验结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 性能对比箱线图
    ax1 = axes[0]
    df = pd.DataFrame(data, columns=algorithm_names)
    df.boxplot(ax=ax1)
    ax1.set_title("Algorithm Performance Comparison")
    ax1.set_ylabel("Performance Metric")
    ax1.set_xlabel("Algorithms")
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 2. Nemenyi 临界差值图（如果有）
    ax2 = axes[1]
    if test_results and "mean_ranks" in test_results:
        mean_ranks = np.array(test_results["mean_ranks"])
        cd = test_results.get("critical_difference", 0)

        # 排序
        sorted_idx = np.argsort(mean_ranks)
        sorted_names = [algorithm_names[i] for i in sorted_idx]
        sorted_ranks = mean_ranks[sorted_idx]

        # 绘制平均秩次
        y_pos = np.arange(len(sorted_names))
        ax2.barh(y_pos, sorted_ranks, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_names)
        ax2.set_xlabel("Average Rank")
        ax2.set_title(f"Nemenyi Test (CD={cd:.3f})")
        ax2.axvline(
            sorted_ranks[0] + cd, color="r", linestyle="--", label=f"CD={cd:.3f}"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="x")
    else:
        ax2.text(
            0.5,
            0.5,
            "No post-hoc test results",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Post-hoc Test")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n可视化结果已保存至: {output_path}")
    plt.close()

    # 3. 成对比较热图（如果有）
    if test_results and "pairwise_pvalues" in test_results:
        p_matrix = np.array(test_results["pairwise_pvalues"])

        plt.figure(figsize=(8, 7))

        # 将 p 值转换为显著性标记
        sig_matrix = np.zeros_like(p_matrix)
        sig_matrix[p_matrix < 0.001] = 3  # p < 0.001
        sig_matrix[(p_matrix >= 0.001) & (p_matrix < 0.01)] = 2  # p < 0.01
        sig_matrix[(p_matrix >= 0.01) & (p_matrix < 0.05)] = 1  # p < 0.05

        sns.heatmap(
            sig_matrix,
            annot=p_matrix,
            fmt=".4f",
            xticklabels=algorithm_names,
            yticklabels=algorithm_names,
            cmap="RdYlGn_r",
            cbar_kws={"label": "Significance"},
            vmin=0,
            vmax=3,
        )

        plt.title("Pairwise Comparison P-values\n" "(Darker = More Significant)")
        plt.tight_layout()

        heatmap_path = output_path.replace(".png", "_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        print(f"热图已保存至: {heatmap_path}")
        plt.close()


def run_hypothesis_testing(
    summary_files: List[str],
    output_dir: str,
    metrics: List[str] = None,
    alpha: float = 0.05,
    test_type: str = "friedman",
):
    """运行完整的假设检验流程"""
    if metrics is None:
        metrics = ["R2", "RMSE", "Accuracy", "F1"]

    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for metric in metrics:
        print(f"\n{'#'*60}")
        print(f"# 指标: {metric}")
        print(f"{'#'*60}")

        # 收集所有实验的数据
        all_data = []
        all_names = None

        for summary_file in summary_files:
            if not os.path.exists(summary_file):
                print(f"警告: 文件不存在 - {summary_file}")
                continue

            try:
                # 尝试从回归或分类结果中提取
                for scenario_key in ["test_reg", "test_cls"]:
                    try:
                        data, names = load_results_from_summary(
                            summary_file, metric, scenario_key
                        )
                        if data.size > 0:
                            all_data.append(data)
                            if all_names is None:
                                all_names = names
                            break
                    except:
                        continue
            except Exception as e:
                print(f"加载数据失败 {summary_file}: {e}")

        if not all_data or all_names is None:
            print(f"指标 {metric} 无可用数据，跳过")
            continue

        # 合并数据
        data_matrix = np.vstack(all_data)

        # 执行假设检验
        test_results = perform_hypothesis_test(data_matrix, all_names, alpha, test_type)

        all_results[metric] = test_results

        # 可视化
        vis_path = os.path.join(output_dir, f"hypothesis_test_{metric}.png")
        visualize_results(data_matrix, all_names, vis_path, test_results)

        # 保存详细结果
        result_path = os.path.join(output_dir, f"hypothesis_test_{metric}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)

    # 保存汇总
    summary_path = os.path.join(output_dir, "hypothesis_testing_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n假设检验完成！结果已保存至: {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="假设检验 - 评估算法显著性差异")
    parser.add_argument(
        "--summaries",
        type=str,
        nargs="+",
        required=True,
        help="实验汇总 JSON 文件路径列表",
    )
    parser.add_argument(
        "--output", type=str, default="./outputs/hypothesis_testing", help="输出目录"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["R2", "RMSE", "MAE", "Accuracy", "F1", "F1_macro"],
        help="要测试的指标列表",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平")
    parser.add_argument(
        "--test",
        type=str,
        default="friedman",
        choices=["friedman", "kruskal"],
        help="检验类型",
    )

    args = parser.parse_args()

    run_hypothesis_testing(
        args.summaries, args.output, args.metrics, args.alpha, args.test
    )


if __name__ == "__main__":
    main()
