"""
模型组件消融实验
测试不同模型组件对性能的影响
"""

import os
import json
import argparse
from typing import Dict, Any, List
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


def run_ablation_experiment(
    config_path: str, ablation_type: str, output_dir: str
) -> Dict[str, Any]:
    """
    运行消融实验

    Args:
        config_path: 配置文件路径
        ablation_type: 消融类型，支持：
            - no_dropout: 移除dropout
            - shallow_tower: 浅层塔（减少tower层数）
            - small_embedding: 小embedding维度
            - fewer_experts: 减少专家数量（仅MMoE）
            - single_expert: 单专家（仅MMoE）
        output_dir: 输出目录

    Returns:
        实验结果字典
    """
    # 加载配置
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # 根据消融类型修改配置
    model_cfg = cfg.get("model", {}).copy()
    original_cfg = model_cfg.copy()

    if ablation_type == "no_dropout":
        model_cfg["dropout"] = 0.0
        desc = "无Dropout"
    elif ablation_type == "shallow_tower":
        model_cfg["tower_hidden_reg"] = [32]
        model_cfg["tower_hidden_cls"] = [32]
        desc = "浅层塔"
    elif ablation_type == "small_embedding":
        model_cfg["embedding_dim"] = 4
        desc = "小Embedding维度"
    elif ablation_type == "fewer_experts":
        if model_cfg.get("type") == "mmoe":
            model_cfg["experts"] = 2
            desc = "减少专家数量"
        else:
            print("fewer_experts仅适用于MMoE模型")
            return None
    elif ablation_type == "single_expert":
        if model_cfg.get("type") == "mmoe":
            model_cfg["experts"] = 1
            desc = "单专家"
        else:
            print("single_expert仅适用于MMoE模型")
            return None
    else:
        raise ValueError(f"未知的消融类型: {ablation_type}")

    # 准备数据
    dcfg = DataConfig(**cfg["dataset"])
    training = cfg.get("training", {})

    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    epochs = int(training.get("epochs", 20))
    lr = float(training.get("lr", 1e-3))
    weight_decay = float(training.get("weight_decay", 0.0))

    train_loader, val_loader, test_loader, meta = prepare_dataloaders(
        dcfg, batch_size=batch_size, num_workers=num_workers, seed=seed
    )

    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    best_test_results = None

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch}/{epochs} [{desc}]", leave=False
        ):
            x_num, x_cat, y_reg, y_cls = to_device(batch, device)
            optimizer.zero_grad()
            y_reg_pred, y_cls_prob = model(x_num, x_cat)
            loss_reg = criterion_reg(y_reg_pred, y_reg)
            loss_cls = criterion_cls(y_cls_prob, y_cls)
            loss = loss_reg + loss_cls
            loss.backward()
            optimizer.step()

        # 验证
        with torch.no_grad():
            val_loss, val_reg_m, val_cls_m = run_epoch(
                model, val_loader, device, criterion_reg, criterion_cls
            )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with torch.no_grad():
                test_loss, test_reg_m, test_cls_m = run_epoch(
                    model, test_loader, device, criterion_reg, criterion_cls
                )
            best_test_results = {
                "test_loss": test_loss,
                "test_reg": test_reg_m,
                "test_cls": test_cls_m,
            }

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "ablation_type": ablation_type,
        "description": desc,
        "original_config": original_cfg,
        "modified_config": model_cfg,
        "results": best_test_results,
    }

    output_file = os.path.join(output_dir, f"{ablation_type}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n{desc} 消融实验完成:")
    print(f"  测试损失: {best_test_results['test_loss']:.4f}")
    print(f"  回归 R2: {best_test_results['test_reg']['R2']:.4f}")
    print(f"  分类 F1: {best_test_results['test_cls']['F1']:.4f}")

    return result


def run_all_ablations(config_path: str, output_dir: str, model_type: str = "mmoe"):
    """运行所有消融实验"""
    print(f"开始运行 {model_type.upper()} 模型的消融实验...\n")

    # 基础消融实验（适用于所有模型）
    base_ablations = ["no_dropout", "shallow_tower", "small_embedding"]

    # MMoE特有的消融实验
    mmoe_ablations = ["fewer_experts", "single_expert"]

    ablations = base_ablations
    if model_type == "mmoe":
        ablations += mmoe_ablations

    results = {}
    for ablation in ablations:
        print(f"\n{'='*50}")
        print(f"运行消融实验: {ablation}")
        print(f"{'='*50}")
        result = run_ablation_experiment(config_path, ablation, output_dir)
        if result:
            results[ablation] = result

    # 保存汇总结果
    summary_file = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print("所有消融实验完成！")
    print(f"结果已保存到: {output_dir}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="模型组件消融实验")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--ablation",
        type=str,
        default="all",
        help="消融类型: no_dropout, shallow_tower, small_embedding, fewer_experts, single_expert, all",
    )
    parser.add_argument(
        "--model_type", type=str, default="mmoe", help="模型类型: mmoe 或 shared_bottom"
    )

    args = parser.parse_args()

    if args.ablation == "all":
        run_all_ablations(args.config, args.output, args.model_type)
    else:
        run_ablation_experiment(args.config, args.ablation, args.output)


if __name__ == "__main__":
    main()
