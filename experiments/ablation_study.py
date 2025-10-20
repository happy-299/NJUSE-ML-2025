"""
模型组件消融实验
"""

import os
import json
import argparse
from typing import Dict, Any, List
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from utils.data_processing import DataConfig, prepare_dataloaders
from models.model import MTLModel
from utils.metrics import regression_metrics, classification_metrics
from main import set_seed, to_device, run_epoch


class AblationConfig:
    """消融实验配置"""

    # 消融实验变体
    VARIANTS = {
        "full": {
            "name": "完整模型",
            "description": "完整的多任务学习模型",
            "modifications": {},
        },
        "no_shared": {
            "name": "无共享层",
            "description": "移除共享底层，每个任务独立建模",
            "modifications": {"bottom_mlp": []},
        },
        "single_tower": {
            "name": "单塔结构",
            "description": "回归和分类使用相同的塔结构",
            "modifications": {"tower_hidden_cls": None},  # 使用与reg相同
        },
        "no_expert": {
            "name": "单专家MMoE",
            "description": "MMoE模型只使用1个专家（退化为共享底层）",
            "modifications": {"experts": 1},
        },
        "shallow": {
            "name": "浅层网络",
            "description": "减少网络深度",
            "modifications": {
                "bottom_mlp": [64],
                "expert_hidden": [32],
                "tower_hidden_reg": [32],
                "tower_hidden_cls": [32],
            },
        },
        "no_dropout": {
            "name": "无Dropout",
            "description": "移除所有Dropout层",
            "modifications": {"dropout": 0.0},
        },
    }


def build_ablation_model(
    meta: Dict[str, Any], base_cfg: Dict[str, Any], variant: str
) -> MTLModel:
    """构建消融实验模型"""
    model_cfg = base_cfg.copy()
    modifications = AblationConfig.VARIANTS[variant]["modifications"]

    # 应用修改
    for key, value in modifications.items():
        if key == "tower_hidden_cls" and value is None:
            # 使用与reg相同的塔结构
            model_cfg[key] = model_cfg.get("tower_hidden_reg", [64, 32])
        else:
            model_cfg[key] = value

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
    )
    return model


def train_ablation_variant(
    config_path: str, variant: str, output_dir: str, seed: int = 42
):
    """训练单个消融实验变体"""
    print(f"\n{'='*60}")
    print(f"消融实验变体: {AblationConfig.VARIANTS[variant]['name']}")
    print(f"描述: {AblationConfig.VARIANTS[variant]['description']}")
    print(f"{'='*60}\n")

    # 加载配置
    cfg = load_config(config_path)
    set_seed(seed)

    # 准备数据
    dcfg = DataConfig(**cfg["dataset"])
    training = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    epochs = int(training.get("epochs", 20))
    lr = float(training.get("lr", 1e-3))
    weight_decay = float(training.get("weight_decay", 0.0))
    patience = int(training.get("patience", 5))

    train_loader, val_loader, test_loader, meta = prepare_dataloaders(
        dcfg, batch_size=batch_size, num_workers=num_workers, seed=seed
    )

    # 构建消融模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ablation_model(meta, model_cfg, variant).to(device)

    # 损失函数
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.BCELoss()

    loss_w = training.get("loss_weights", {"reg": 1.0, "cls": 1.0})
    w_reg = float(loss_w.get("reg", 1.0))
    w_cls = float(loss_w.get("cls", 1.0))

    def combined_loss(y_reg_pred, y_reg, y_cls_prob, y_cls):
        return w_reg * criterion_reg(y_reg_pred, y_reg) + w_cls * criterion_cls(
            y_cls_prob, y_cls
        )

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 输出路径
    variant_dir = os.path.join(output_dir, f"ablation_{variant}")
    os.makedirs(variant_dir, exist_ok=True)
    best_path = os.path.join(variant_dir, "best.pt")

    # 训练
    best_metric = float("inf")
    best_epoch = -1
    patience_left = patience
    history = []

    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x_num, x_cat, y_reg, y_cls = to_device(batch, device)
            optimizer.zero_grad()
            y_reg_pred, y_cls_prob = model(x_num, x_cat)
            loss = combined_loss(y_reg_pred, y_reg, y_cls_prob, y_cls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y_reg.size(0)

        train_loss = total_loss / max(1, len(train_loader.dataset))

        # 验证
        with torch.no_grad():
            val_loss, val_reg_m, val_cls_m = run_epoch(
                model, val_loader, device, criterion_reg, criterion_cls
            )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_reg": val_reg_m,
            "val_cls": val_cls_m,
        }
        history.append(record)

        # 早停
        if val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch
            patience_left = patience
            torch.save({"model_state": model.state_dict(), "meta": meta}, best_path)
        else:
            patience_left -= 1

        if patience_left <= 0:
            print(f"早停于 Epoch {epoch}")
            break

    # 测试
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    with torch.no_grad():
        test_loss, test_reg_m, test_cls_m = run_epoch(
            model, test_loader, device, criterion_reg, criterion_cls
        )

    # 保存结果
    results = {
        "variant": variant,
        "name": AblationConfig.VARIANTS[variant]["name"],
        "description": AblationConfig.VARIANTS[variant]["description"],
        "best_epoch": best_epoch,
        "val_best": best_metric,
        "test_loss": test_loss,
        "test_reg": test_reg_m,
        "test_cls": test_cls_m,
    }

    with open(os.path.join(variant_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(variant_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n变体 {variant} 测试结果:")
    print(
        f"  回归 - R2: {test_reg_m['R2']:.4f}, RMSE: {test_reg_m['RMSE']:.4f}, MAE: {test_reg_m['MAE']:.4f}"
    )
    print(
        f"  分类 - Acc: {test_cls_m['Accuracy']:.4f}, F1: {test_cls_m['F1']:.4f}, F1_macro: {test_cls_m['F1_macro']:.4f}"
    )

    return results


def run_ablation_study(
    config_path: str, output_dir: str, variants: List[str] = None, seed: int = 42
):
    """运行完整的消融实验"""
    if variants is None:
        variants = list(AblationConfig.VARIANTS.keys())

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for variant in variants:
        try:
            result = train_ablation_variant(config_path, variant, output_dir, seed)
            all_results.append(result)
        except Exception as e:
            print(f"变体 {variant} 训练失败: {e}")

    # 保存汇总结果
    summary_path = os.path.join(output_dir, "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n消融实验完成！结果已保存至: {output_dir}")
    print(f"汇总报告: {summary_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="模型组件消融实验")
    parser.add_argument("--config", type=str, required=True, help="基础配置文件路径")
    parser.add_argument(
        "--output", type=str, default="./outputs/ablation", help="输出目录"
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=None,
        help="要测试的变体列表，默认全部",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    run_ablation_study(args.config, args.output, args.variants, args.seed)


if __name__ == "__main__":
    main()
