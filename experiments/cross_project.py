"""
跨项目泛化性实验
测试模型在不同项目间的泛化能力
"""

import os
import json
import argparse
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing import DataConfig, prepare_dataloaders
from models.model import MTLModel
from utils.metrics import regression_metrics, classification_metrics
from config import load_config
from main import set_seed, to_device, run_epoch


def train_on_project(
    train_projects: List[str], test_project: str, config_path: str, output_dir: str
) -> Dict[str, Any]:
    """
    在指定项目上训练并在另一个项目上测试

    Args:
        train_projects: 训练项目列表（engineered目录下的文件名，不含扩展名）
        test_project: 测试项目名
        config_path: 配置文件路径
        output_dir: 输出目录

    Returns:
        实验结果字典
    """
    # 加载配置
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    training = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    epochs = int(training.get("epochs", 20))
    lr = float(training.get("lr", 1e-3))
    weight_decay = float(training.get("weight_decay", 0.0))
    patience = int(training.get("patience", 5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 准备训练数据
    print(f"\n准备训练数据: {', '.join(train_projects)}")
    train_loaders = []
    val_loaders = []
    metas = []

    for project in train_projects:
        dcfg_dict = cfg["dataset"].copy()
        dcfg_dict["file"] = f"./engineered/{project}_engineered.xlsx"
        dcfg = DataConfig(**dcfg_dict)

        train_loader, val_loader, _, curr_meta = prepare_dataloaders(
            dcfg, batch_size=batch_size, num_workers=num_workers, seed=seed
        )
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

        # 收集每个项目的meta以便后续合并
        metas.append(curr_meta)

    # 准备测试数据
    print(f"准备测试数据: {test_project}")
    dcfg_test_dict = cfg["dataset"].copy()
    dcfg_test_dict["file"] = f"./engineered/{test_project}_engineered.xlsx"
    dcfg_test = DataConfig(**dcfg_test_dict)

    _, _, test_loader, test_meta = prepare_dataloaders(
        dcfg_test, batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    # 将测试集的meta也加入用于合并
    metas.append(test_meta)

    # 合并meta，确保cat_cardinalities覆盖所有项目（取每列的最大值）
    # 假设所有数据集的numeric_dim一致，类别特征数也应一致；若不一致以较大长度为准并填充0
    numeric_dim = metas[0].get("numeric_dim", 0)
    # 合并类别基数
    cat_lists = [m.get("cat_cardinalities", []) for m in metas]
    max_len = max(len(lst) for lst in cat_lists)
    padded = [lst + [0] * (max_len - len(lst)) for lst in cat_lists]
    combined_cardinalities = [int(max(col)) for col in zip(*padded)]

    # 合并其它字段（如cat_onehot_dim）: 取最大的
    cat_onehot_dim = max(m.get("cat_onehot_dim", 0) for m in metas)

    # 构造最终meta
    meta = {
        "numeric_dim": numeric_dim,
        "cat_cardinalities": combined_cardinalities,
        "cat_onehot_dim": cat_onehot_dim,
    }

    # 构建模型（使用合并后的meta）
    model = MTLModel(
        numeric_dim=meta["numeric_dim"],
        cat_cardinalities=meta["cat_cardinalities"],
        cat_onehot_dim=meta["cat_onehot_dim"],
        embedding_dim=int(model_cfg.get("embedding_dim", 16)),
        bottom_mlp=list(model_cfg.get("bottom_mlp", [128, 64])),
        model_type=str(model_cfg.get("type", "mmoe")),
        experts=int(model_cfg.get("experts", 4)),
        expert_hidden=list(model_cfg.get("expert_hidden", [64])),
        tower_hidden_reg=list(model_cfg.get("tower_hidden_reg", [64, 32])),
        tower_hidden_cls=list(model_cfg.get("tower_hidden_cls", [64, 32])),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device)

    # 损失函数和优化器
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 训练
    best_val_loss = float("inf")
    patience_left = patience
    history = []

    for epoch in range(1, epochs + 1):
        # 训练（轮流使用不同项目的数据）
        model.train()
        total_loss = 0.0
        total_samples = 0

        for train_loader in train_loaders:
            for batch in tqdm(
                train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False
            ):
                x_num, x_cat, y_reg, y_cls = to_device(batch, device)
                optimizer.zero_grad()
                y_reg_pred, y_cls_prob = model(x_num, x_cat)
                loss_reg = criterion_reg(y_reg_pred, y_reg)
                loss_cls = criterion_cls(y_cls_prob, y_cls)
                loss = loss_reg + loss_cls
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y_reg.size(0)
                total_samples += y_reg.size(0)

        train_loss = total_loss / max(1, total_samples)

        # 验证（在所有训练项目的验证集上）
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for val_loader in val_loaders:
                for batch in val_loader:
                    x_num, x_cat, y_reg, y_cls = to_device(batch, device)
                    y_reg_pred, y_cls_prob = model(x_num, x_cat)
                    loss_reg = criterion_reg(y_reg_pred, y_reg)
                    loss_cls = criterion_cls(y_cls_prob, y_cls)
                    loss = loss_reg + loss_cls
                    val_loss_sum += loss.item() * y_reg.size(0)
                    val_samples += y_reg.size(0)

        val_loss = val_loss_sum / max(1, val_samples)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = patience
            # 保存最佳模型
            best_model_state = model.state_dict()
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"早停于 Epoch {epoch}")
                break

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        test_loss, test_reg_m, test_cls_m = run_epoch(
            model, test_loader, device, criterion_reg, criterion_cls
        )

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "train_projects": train_projects,
        "test_project": test_project,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_reg": test_reg_m,
        "test_cls": test_cls_m,
        "history": history,
    }

    output_file = os.path.join(
        output_dir, f"train_{'_'.join(train_projects)}_test_{test_project}.json"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n跨项目实验完成:")
    print(f"  训练项目: {', '.join(train_projects)}")
    print(f"  测试项目: {test_project}")
    print(f"  测试损失: {test_loss:.4f}")
    print(f"  回归 R2: {test_reg_m['R2']:.4f}")
    print(f"  分类 Accuracy: {test_cls_m['Accuracy']:.4f}")
    print(f"  分类 F1: {test_cls_m['F1']:.4f}")

    return result


def run_cross_project_experiments(
    projects: List[str], config_path: str, output_dir: str
):
    """
    运行完整的跨项目实验

    Args:
        projects: 项目列表
        config_path: 配置文件路径
        output_dir: 输出目录
    """
    print(f"开始跨项目泛化性实验...")
    print(f"项目列表: {', '.join(projects)}\n")

    all_results = []

    # 对每个项目，使用其他所有项目训练，在该项目上测试
    for i, test_project in enumerate(projects):
        train_projects = [p for j, p in enumerate(projects) if j != i]

        print(f"\n{'='*60}")
        print(f"实验 {i+1}/{len(projects)}")
        print(f"{'='*60}")

        result = train_on_project(train_projects, test_project, config_path, output_dir)
        all_results.append(result)

    # 保存汇总结果
    summary = {
        "projects": projects,
        "results": all_results,
        "summary_statistics": compute_summary_stats(all_results),
    }

    summary_file = os.path.join(output_dir, "cross_project_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("所有跨项目实验完成！")
    print(f"结果已保存到: {output_dir}")
    print(f"{'='*60}")

    # 打印汇总统计
    print_summary_stats(summary["summary_statistics"])


def compute_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算汇总统计信息"""
    r2_scores = [r["test_reg"]["R2"] for r in results]
    rmse_scores = [r["test_reg"]["RMSE"] for r in results]
    mae_scores = [r["test_reg"]["MAE"] for r in results]
    acc_scores = [r["test_cls"]["Accuracy"] for r in results]
    f1_scores = [r["test_cls"]["F1"] for r in results]

    return {
        "regression": {
            "R2_mean": float(np.mean(r2_scores)),
            "R2_std": float(np.std(r2_scores)),
            "RMSE_mean": float(np.mean(rmse_scores)),
            "RMSE_std": float(np.std(rmse_scores)),
            "MAE_mean": float(np.mean(mae_scores)),
            "MAE_std": float(np.std(mae_scores)),
        },
        "classification": {
            "Accuracy_mean": float(np.mean(acc_scores)),
            "Accuracy_std": float(np.std(acc_scores)),
            "F1_mean": float(np.mean(f1_scores)),
            "F1_std": float(np.std(f1_scores)),
        },
    }


def print_summary_stats(stats: Dict[str, Any]):
    """打印汇总统计信息"""
    print("\n跨项目实验汇总统计:")
    print("\n回归指标:")
    print(
        f"  R2: {stats['regression']['R2_mean']:.4f} ± {stats['regression']['R2_std']:.4f}"
    )
    print(
        f"  RMSE: {stats['regression']['RMSE_mean']:.2f} ± {stats['regression']['RMSE_std']:.2f}"
    )
    print(
        f"  MAE: {stats['regression']['MAE_mean']:.2f} ± {stats['regression']['MAE_std']:.2f}"
    )
    print("\n分类指标:")
    print(
        f"  Accuracy: {stats['classification']['Accuracy_mean']:.4f} ± {stats['classification']['Accuracy_std']:.4f}"
    )
    print(
        f"  F1: {stats['classification']['F1_mean']:.4f} ± {stats['classification']['F1_std']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="跨项目泛化性实验")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--projects",
        type=str,
        nargs="+",
        default=["django", "react", "tensorflow"],
        help="项目列表（不含_engineered.xlsx后缀）",
    )

    args = parser.parse_args()

    run_cross_project_experiments(args.projects, args.config, args.output)


if __name__ == "__main__":
    main()
