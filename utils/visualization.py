import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple


def plot_training_history(history_path: str, save_dir: Optional[str] = None):
    """
    绘制训练历史曲线

    Args:
        history_path: 训练历史JSON文件路径
        save_dir: 图表保存目录，如果为None则显示而不保存
    """
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    val_loss = [record["val_loss"] for record in history]

    # 获取验证集指标
    val_reg_metrics = {}
    val_cls_metrics = {}

    for metric in ["MAE", "MSE", "RMSE", "R2"]:
        if metric in history[0]["val_reg"]:
            val_reg_metrics[metric] = [record["val_reg"][metric] for record in history]

    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        if metric in history[0]["val_cls"]:
            val_cls_metrics[metric] = [record["val_cls"][metric] for record in history]

    # 创建子图
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Loss curves
    axs[0, 0].plot(epochs, train_loss, "b-", label="Train Loss")
    axs[0, 0].plot(epochs, val_loss, "r-", label="Val Loss")
    axs[0, 0].set_title("Training and Validation Loss")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Regression metrics
    for i, (metric, values) in enumerate(val_reg_metrics.items()):
        axs[0, 1].plot(epochs, values, label=metric)
    axs[0, 1].set_title("Regression Metrics")
    axs[0, 1].set_xlabel("Epochs")
    axs[0, 1].set_ylabel("Value")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Classification metrics
    for i, (metric, values) in enumerate(val_cls_metrics.items()):
        axs[1, 0].plot(epochs, values, label=metric)
    axs[1, 0].set_title("Classification Metrics")
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "training_history.png"))
        plt.close()
    else:
        plt.show()


def plot_model_comparison(
    results_dirs: List[str], model_names: List[str], save_path: Optional[str] = None
):
    """
    比较不同模型的性能

    Args:
        results_dirs: 结果目录列表
        model_names: 模型名称列表
        save_path: 图表保存路径，如果为None则显示而不保存
    """
    results = []

    for model_dir in results_dirs:
        with open(os.path.join(model_dir, "results.json"), "r") as f:
            result = json.load(f)
            results.append(result)

    # 准备数据
    reg_metrics = ["MAE", "MSE", "RMSE", "R2"]
    cls_metrics = ["Accuracy", "Precision", "Recall", "F1"]

    reg_data = {
        metric: [r["test_reg"][metric] for r in results] for metric in reg_metrics
    }
    cls_data = {
        metric: [r["test_cls"][metric] for r in results] for metric in cls_metrics
    }

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    width = 0.15
    x = np.arange(len(model_names))

    # Regression metrics
    for i, (metric, values) in enumerate(reg_data.items()):
        ax1.bar(x + (i - 1.5) * width, values, width, label=metric)

    ax1.set_title("Regression Performance Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.legend()
    ax1.grid(True, axis="y")

    # Classification metrics
    for i, (metric, values) in enumerate(cls_data.items()):
        ax2.bar(x + (i - 1.5) * width, values, width, label=metric)

    ax2.set_title("Classification Performance Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(True, axis="y")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
