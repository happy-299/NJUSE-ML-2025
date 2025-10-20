import os
import random
import json
import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from utils.data_processing import DataConfig, prepare_dataloaders
from models.model import MTLModel
from utils.metrics import regression_metrics, classification_metrics
from config import load_config, DATA_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR


def set_seed(seed: int):
    """设置随机种子以确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_device(batch, device: torch.device):
    """将数据批次移至指定设备"""
    x_num, x_cat, y_reg, y_cls = batch
    x_num = x_num.to(device) if x_num is not None and x_num.numel() > 0 else None
    x_cat = x_cat.to(device) if x_cat is not None and x_cat.numel() > 0 else None
    return x_num, x_cat, y_reg.to(device), y_cls.to(device)


def build_model(meta: Dict[str, Any], model_cfg: Dict[str, Any]) -> MTLModel:
    """构建模型"""
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


def run_epoch(model, loader, device, criterion_reg, criterion_cls, optimizer=None):
    """运行一个训练/评估周期"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    y_reg_list, y_reg_pred_list = [], []
    y_cls_list, y_cls_prob_list = [], []

    for batch in tqdm(loader, desc="训练中" if is_train else "评估中", leave=False):
        x_num, x_cat, y_reg, y_cls = to_device(batch, device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            y_reg_pred, y_cls_prob = model(x_num, x_cat)
            loss_reg = criterion_reg(y_reg_pred, y_reg)
            loss_cls = criterion_cls(y_cls_prob, y_cls)
            loss = loss_reg + loss_cls
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * y_reg.size(0)
        y_reg_list.append(y_reg.detach().cpu().numpy())
        y_reg_pred_list.append(y_reg_pred.detach().cpu().numpy())
        y_cls_list.append(y_cls.detach().cpu().numpy())
        y_cls_prob_list.append(y_cls_prob.detach().cpu().numpy())

    n = len(loader.dataset)
    avg_loss = total_loss / max(1, n)
    y_reg_true = np.vstack(y_reg_list)
    y_reg_pred = np.vstack(y_reg_pred_list)
    y_cls_true = np.vstack(y_cls_list)
    y_cls_prob = np.vstack(y_cls_prob_list)

    reg_metrics_values = regression_metrics(y_reg_true, y_reg_pred)
    cls_metrics_values = classification_metrics(y_cls_true.astype(int), y_cls_prob)
    return avg_loss, reg_metrics_values, cls_metrics_values


def train_and_eval(config_path: str):
    """训练和评估模型"""
    # 加载配置
    cfg = load_config(config_path)

    # 设置随机种子
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # 准备数据
    dcfg = DataConfig(**cfg["dataset"])
    training = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    out_cfg = cfg.get("output", {})

    batch_size = int(training.get("batch_size", 256))
    num_workers = int(training.get("num_workers", 0))
    epochs = int(training.get("epochs", 20))
    lr = float(training.get("lr", 1e-3))
    weight_decay = float(training.get("weight_decay", 0.0))
    patience = int(training.get("patience", 5))

    train_loader, val_loader, test_loader, meta = prepare_dataloaders(
        dcfg, batch_size=batch_size, num_workers=num_workers, seed=seed
    )

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(meta, model_cfg).to(device)

    # 损失函数
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.BCELoss()

    # 可选损失权重
    loss_w = training.get("loss_weights", {"reg": 1.0, "cls": 1.0})
    w_reg = float(loss_w.get("reg", 1.0))
    w_cls = float(loss_w.get("cls", 1.0))

    # 将权重融合到 criterion
    def combined_loss(y_reg_pred, y_reg, y_cls_prob, y_cls):
        return w_reg * criterion_reg(y_reg_pred, y_reg) + w_cls * criterion_cls(y_cls_prob, y_cls)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 输出目录设置
    out_dir = out_cfg.get("dir", OUTPUTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(CHECKPOINTS_DIR, "best.pt")
    history_path = os.path.join(out_dir, "history.json")

    # 训练参数初始化
    best_metric = float("inf")
    best_epoch = -1
    patience_left = patience
    history = []

    # 训练循环
    for epoch in range(1, epochs + 1):
        # 训练一个epoch
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
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
            val_loss, val_reg_m, val_cls_m = run_epoch(model, val_loader, device, criterion_reg, criterion_cls)

        # 记录训练历史
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_reg": val_reg_m,
            "val_cls": val_cls_m,
        }
        history.append(record)

        # 早停监控
        if val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch
            patience_left = patience
            torch.save({"model_state": model.state_dict(), "meta": meta, "cfg": cfg}, best_path)
        else:
            patience_left -= 1

        # 保存历史
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        if patience_left <= 0:
            print(f"早停：Epoch {epoch}（最佳：Epoch {best_epoch}）")
            break

    # 测试集评估（加载best模型）
    if os.path.exists(best_path):
        try:
            ckpt = torch.load(best_path, map_location=device)
        except TypeError:
            ckpt = torch.load(best_path, map_location=device)

        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)

    with torch.no_grad():
        test_loss, test_reg_m, test_cls_m = run_epoch(model, test_loader, device, criterion_reg, criterion_cls)

    # 保存结果
    results = {
        "best_epoch": best_epoch,
        "val_best": best_metric,
        "test_loss": test_loss,
        "test_reg": test_reg_m,
        "test_cls": test_cls_m,
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("测试结果:", json.dumps(results, ensure_ascii=False, indent=2))


def main():
    """主函数：解析命令行参数并启动训练"""
    parser = argparse.ArgumentParser(description="多任务学习模型训练")
    parser.add_argument("--config", type=str, default="./configs/example.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    train_and_eval(args.config)


if __name__ == "__main__":
    main()