import os
import yaml
import json
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 根据 EXPERIMENT_REPORT.md 定义完整的特征分组
# 注意：这里的特征名需要与你的数据文件中的列名完全一致
FEATURE_GROUPS = {
    "PR基本信息": [
        "number", "state", "title", "author", "body", "created_at",
        "updated_at", "merged_at", "merged", "comments", "review_comments",
        "commits"
    ],
    "代码变更": [
        "additions", "deletions", "changed_files", "lines_added",
        "lines_deleted", "segs_added", "segs_deleted", "segs_updated",
        "files_added", "files_deleted", "files_updated", "modify_proportion",
        "modify_entropy", "test_churn", "non_test_churn"
    ],
    "文本语义": [
        "title_length", "title_readability", "title_embedding", "body_length",
        "body_readability", "body_embedding", "comment_length",
        "comment_embedding", "avg_comment_length"
    ],
    "评审协作": [
        "reviewer_num", "bot_reviewer_num", "is_reviewed", "comment_num",
        "last_comment_mention", "reviewer_count", "avg_reviewers", "avg_rounds"
    ],
    "作者经验": [
        "name", "experience", "is_reviewer", "change_num", "participation",
        "changes_per_week", "avg_round", "avg_duration", "merge_proportion",
        "author_experience", "author_activity", "author_exp_change_size"
    ],
    "网络中心性": [
        "degree_centrality", "closeness_centrality", "betweenness_centrality",
        "eigenvector_centrality", "clustering_coefficient", "k_coreness",
        "reviewer_degree_centrality", "reviewer_closeness_centrality",
        "reviewer_betweenness_centrality", "reviewer_eigenvector_centrality",
        "reviewer_clustering_coefficient", "reviewer_k_coreness"
    ],
    "项目上下文": [
        "project_age", "open_changes", "author_num", "team_size",
        "changes_per_author", "changes_per_reviewer", "avg_lines", "avg_segs",
        "add_per_week", "del_per_week"
    ],
    "时间模式": [
        "created_hour", "created_dayofweek", "created_month",
        "processing_time", "last_response_time", "last_comment_time"
    ],
    "语义标签": [
        "has_test", "has_feature", "has_bug", "has_document", "has_improve",
        "has_refactor", "has_bug_keyword", "has_feature_keyword",
        "has_document_keyword"
    ],
    "复杂度指标": [
        "total_changes", "net_changes", "change_density", "additions_per_file",
        "complexity_per_reviewer"
    ],
}


def run_single_experiment(base_config: dict,
                          temp_config_path: str,
                          exclude_cols: list = None):
    """
    运行一次单独的训练评估实验。

    Args:
        base_config (dict): 基础配置字典。
        temp_config_path (str): 临时配置文件的保存路径。
        exclude_cols (list, optional): 需要排除的特征列。Defaults to None.

    Returns:
        float: 分类任务的准确率。
    """
    config = base_config.copy()

    # 注入要排除的列
    if 'dataset' not in config:
        config['dataset'] = {}
    config['dataset']['exclude_cols'] = exclude_cols if exclude_cols else []

    # 写入临时配置文件
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)

    # 运行主脚本
    os.system(f"python main.py --config {temp_config_path}")

    # 从结果文件中读取指标
    output_dir = config.get('output', {}).get('dir', './outputs/default')
    results_path = os.path.join(output_dir, 'results.json')

    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)

        reg = results_data.get('test_reg', {})
        cls = results_data.get('test_cls', {})

        return reg, cls
    except Exception as e:
        print(f"读取结果失败: {e}")
        return {}, {}


def plot_metrics_table(metrics_dict, metric_keys, title, save_path):
    df = pd.DataFrame(metrics_dict).T[metric_keys]
    print(f"\n{title} (可直接复制到报告):\n")
    print(df.to_markdown())
    df.to_csv(save_path.replace('.png', '.csv'))
    plt.figure(figsize=(12, 7))
    df.plot(kind='barh')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"图表已保存到: {save_path}")
    plt.close()


def main():
    base_config_path = "configs/django_real_mmoe.yaml"
    temp_dir = "configs/temp_ablation"

    # 清理并创建临时配置文件夹
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    project_name = os.path.splitext(os.path.basename(base_config_path))[0]
    reg_results = {}
    cls_results = {}

    # 1. 运行基线实验 (不排除任何特征)
    print("\n========== 1. 开始基线实验 (所有特征) ==========")
    temp_baseline_path = os.path.join(temp_dir, "baseline.yaml")
    reg, cls = run_single_experiment(base_config,
                                     temp_baseline_path,
                                     exclude_cols=[])
    reg_results["所有特征"] = reg
    cls_results["所有特征"] = cls
    print(f"基线分类: {cls}")
    print(f"基线回归: {reg}")

    # 2. 运行特征消融实验
    for i, (group_name, group_features) in enumerate(FEATURE_GROUPS.items()):
        print(f"\n========== {i+2}. 开始消融实验: {group_name} ==========")
        temp_ablation_path = os.path.join(temp_dir,
                                          f"ablation_{group_name}.yaml")
        reg, cls = run_single_experiment(base_config,
                                         temp_ablation_path,
                                         exclude_cols=group_features)
        reg_results[f"禁用-{group_name}"] = reg
        cls_results[f"禁用-{group_name}"] = cls
        print(f"禁用 {group_name} 分类: {cls}")
        print(f"禁用 {group_name} 回归: {reg}")

    # 3. 清理临时文件
    print("\n清理临时配置文件...")
    shutil.rmtree(temp_dir)

    # 4. 打印并绘制最终结果
    print("\n========== 最终消融实验结果表格 ==========")
    reg_keys = ["R2", "RMSE", "MAE", "MSE"]
    cls_keys = ["Accuracy", "Precision", "Recall", "F1", "F1_macro"]
    plot_metrics_table(reg_results, reg_keys, f"{project_name} 回归消融实验结果",
                       f"outputs/figures/{project_name}_reg_ablation.png")
    plot_metrics_table(cls_results, cls_keys, f"{project_name} 分类消融实验结果",
                       f"outputs/figures/{project_name}_cls_ablation.png")


if __name__ == "__main__":
    main()
