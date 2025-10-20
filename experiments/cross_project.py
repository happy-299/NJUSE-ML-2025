"""
跨项目泛化性实验
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from utils.data_processing import DataConfig, MTLDataset, prepare_dataloaders
from models.model import MTLModel
from utils.metrics import regression_metrics, classification_metrics
from main import set_seed, to_device, run_epoch, build_model


class CrossProjectConfig:
    """跨项目实验配置"""

    # 可用的项目数据（需要根据实际数据调整）
    AVAILABLE_PROJECTS = {
        "django": "engineered/django_engineered.xlsx",
        "flask": "engineered/flask_engineered.xlsx",  # 示例
        "requests": "engineered/requests_engineered.xlsx",  # 示例
        "synth": "synth.csv",  # 合成数据作为基准
    }

    # 跨项目实验场景
    SCENARIOS = {
        "within_project": {
            "name": "项目内评估（基线）",
            "description": "在同一项目的训练/测试集上评估",
        },
        "cross_project": {
            "name": "跨项目迁移",
            "description": "在项目A训练，在项目B测试",
        },
        "mixed_training": {
            "name": "混合训练",
            "description": "在多个项目的混合数据上训练，在目标项目测试",
        },
    }


def load_project_data(project_name: str, data_dir: str = "./data") -> pd.DataFrame:
    """加载项目数据"""
    if project_name not in CrossProjectConfig.AVAILABLE_PROJECTS:
        raise ValueError(f"未知项目: {project_name}")

    file_path = os.path.join(
        data_dir, CrossProjectConfig.AVAILABLE_PROJECTS[project_name]
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

    return df


def prepare_cross_project_data(
    train_projects: List[str],
    test_project: str,
    cfg: Dict[str, Any],
    data_dir: str = "./data",
):
    """准备跨项目数据"""
    print(f"\n准备数据: 训练={train_projects}, 测试={test_project}")

    # 加载训练项目数据
    train_dfs = []
    for proj in train_projects:
        df = load_project_data(proj, data_dir)
        df["project"] = proj  # 添加项目标识
        train_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True)

    # 加载测试项目数据
    test_df = load_project_data(test_project, data_dir)
    test_df["project"] = test_project

    print(f"训练样本数: {len(train_df)}, 测试样本数: {len(test_df)}")

    return train_df, test_df


def train_cross_project_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Dict[str, Any],
    output_dir: str,
    seed: int = 42,
):
    """训练跨项目模型"""
    set_seed(seed)

    # 数据配置
    dataset_cfg = cfg.get("dataset", {})
    training = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    batch_size = int(training.get("batch_size", 256))
    epochs = int(training.get("epochs", 20))
    lr = float(training.get("lr", 1e-3))
    weight_decay = float(training.get("weight_decay", 0.0))
    patience = int(training.get("patience", 5))

    # 准备数据集（这里需要根据实际的 DataConfig 调整）
    # 简化版本：使用合成数据的配置
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # 提取特征和标签
    numeric_features = dataset_cfg.get("numeric_features", None)
    categorical_features = dataset_cfg.get("categorical_features", [])
    target_reg = dataset_cfg.get("target_reg", "processing_time")
    target_cls = dataset_cfg.get("target_cls", "merged")

    # 自动检测数值特征
    if numeric_features is None:
        numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [
            f for f in numeric_features if f not in [target_reg, target_cls, "project"]
        ]

    # 构建简化的训练/验证/测试集
    # 注意：这里为了演示简化了实现，实际应该使用 prepare_dataloaders
    print(
        f"数值特征数: {len(numeric_features)}, 类别特征数: {len(categorical_features)}"
    )

    # 由于跨项目场景复杂，这里返回模拟结果
    # 实际实现需要完整的数据处理流程
    results = {
        "train_projects": list(train_df["project"].unique()),
        "test_project": test_df["project"].iloc[0],
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "status": "模拟结果 - 需要完整数据实现",
    }

    return results


def run_within_project_baseline(
    project: str, config_path: str, output_dir: str, seed: int = 42
):
    """运行项目内基线实验"""
    print(f"\n{'='*60}")
    print(f"基线实验: 项目内评估 - {project}")
    print(f"{'='*60}\n")

    cfg = load_config(config_path)

    # 修改配置文件指向特定项目
    # 这里假设配置已经指向正确的数据文件

    variant_dir = os.path.join(output_dir, f"within_{project}")
    os.makedirs(variant_dir, exist_ok=True)

    # 使用原有的训练流程
    from main import train_and_eval

    # 临时修改输出目录
    original_out_dir = cfg["output"]["dir"]
    cfg["output"]["dir"] = variant_dir

    # 保存临时配置
    temp_config = os.path.join(variant_dir, "temp_config.yaml")
    import yaml

    with open(temp_config, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    try:
        train_and_eval(temp_config)

        # 读取结果
        with open(os.path.join(variant_dir, "results.json"), "r") as f:
            results = json.load(f)

        results["scenario"] = "within_project"
        results["project"] = project

        return results
    except Exception as e:
        print(f"基线实验失败: {e}")
        return {"scenario": "within_project", "project": project, "error": str(e)}


def run_cross_project_experiment(
    train_projects: List[str],
    test_project: str,
    config_path: str,
    output_dir: str,
    data_dir: str = "./data",
    seed: int = 42,
):
    """运行跨项目实验"""
    print(f"\n{'='*60}")
    print(f"跨项目实验:")
    print(f"  训练项目: {', '.join(train_projects)}")
    print(f"  测试项目: {test_project}")
    print(f"{'='*60}\n")

    cfg = load_config(config_path)

    variant_name = f"cross_{'_'.join(train_projects)}_to_{test_project}"
    variant_dir = os.path.join(output_dir, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    try:
        # 准备跨项目数据
        train_df, test_df = prepare_cross_project_data(
            train_projects, test_project, cfg, data_dir
        )

        # 训练模型
        results = train_cross_project_model(train_df, test_df, cfg, variant_dir, seed)

        results["scenario"] = "cross_project"
        results["train_projects"] = train_projects
        results["test_project"] = test_project

        # 保存结果
        with open(
            os.path.join(variant_dir, "results.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return results
    except Exception as e:
        print(f"跨项目实验失败: {e}")
        return {
            "scenario": "cross_project",
            "train_projects": train_projects,
            "test_project": test_project,
            "error": str(e),
        }


def run_generalization_study(
    config_path: str, output_dir: str, projects: List[str] = None, seed: int = 42
):
    """运行完整的泛化性研究"""
    if projects is None:
        # 使用可用项目的子集
        projects = ["django"]  # 默认只用 django

    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    # 1. 项目内基线
    for proj in projects:
        try:
            result = run_within_project_baseline(proj, config_path, output_dir, seed)
            all_results.append(result)
        except Exception as e:
            print(f"项目 {proj} 基线实验失败: {e}")

    # 2. 跨项目实验（如果有多个项目）
    if len(projects) > 1:
        for test_proj in projects:
            train_projs = [p for p in projects if p != test_proj]
            try:
                result = run_cross_project_experiment(
                    train_projs, test_proj, config_path, output_dir, "./data", seed
                )
                all_results.append(result)
            except Exception as e:
                print(f"跨项目实验失败 (测试={test_proj}): {e}")

    # 保存汇总
    summary_path = os.path.join(output_dir, "generalization_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n泛化性实验完成！结果已保存至: {output_dir}")
    print(f"汇总报告: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="跨项目泛化性实验")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument(
        "--output", type=str, default="./outputs/cross_project", help="输出目录"
    )
    parser.add_argument(
        "--projects", type=str, nargs="+", default=["django"], help="要使用的项目列表"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    run_generalization_study(args.config, args.output, args.projects, args.seed)


if __name__ == "__main__":
    main()
