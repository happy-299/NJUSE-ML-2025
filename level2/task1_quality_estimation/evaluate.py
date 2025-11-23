"""
Level 2 Task 1: 代码质量评估 - 评估脚本

评估 LLM 推理结果
"""

import os
import sys
import argparse
import json
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import numpy as np

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config import LEVEL2_OUTPUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args):
    """评估预测结果"""
    # 加载预测结果
    predictions_file = LEVEL2_OUTPUT / "task1" / args.predictions_file
    logger.info(f"Loading predictions from {predictions_file}")

    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # 提取预测和真实标签
    y_pred = []
    y_true = []
    confidences = []

    for pred in predictions:
        if pred["ground_truth"] >= 0:  # 有效标签
            y_pred.append(pred["prediction"])
            y_true.append(pred["ground_truth"])
            confidences.append(pred.get("confidence", 0.0))

    if len(y_true) == 0:
        logger.error("No valid samples found")
        return

    logger.info(f"Evaluating {len(y_true)} samples")

    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # 计算各类别的指标
    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    )

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 平均置信度
    avg_confidence = np.mean(confidences)

    # 结果
    results = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "per_class": {
            "class_0": {
                "precision": float(precision_per_class[0]),
                "recall": float(recall_per_class[0]),
                "f1": float(f1_per_class[0]),
                "support": int(support[0]),
            },
            "class_1": {
                "precision": float(precision_per_class[1]),
                "recall": float(recall_per_class[1]),
                "f1": float(f1_per_class[1]),
                "support": int(support[1]),
            },
        },
        "confusion_matrix": cm.tolist(),
        "avg_confidence": float(avg_confidence),
        "total_samples": len(y_true),
    }

    # 打印结果
    logger.info("***** Evaluation Results *****")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision (Macro): {precision:.4f}")
    logger.info(f"  Recall (Macro): {recall:.4f}")
    logger.info(f"  F1-Score (Macro): {f1:.4f}")
    logger.info(f"  Avg Confidence: {avg_confidence:.4f}")
    logger.info(f"\n  Class 0 (No Review):")
    logger.info(f"    Precision: {precision_per_class[0]:.4f}")
    logger.info(f"    Recall: {recall_per_class[0]:.4f}")
    logger.info(f"    F1: {f1_per_class[0]:.4f}")
    logger.info(f"  Class 1 (Needs Review):")
    logger.info(f"    Precision: {precision_per_class[1]:.4f}")
    logger.info(f"    Recall: {recall_per_class[1]:.4f}")
    logger.info(f"    F1: {f1_per_class[1]:.4f}")
    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"    {cm}")

    # 保存结果
    output_file = LEVEL2_OUTPUT / "task1" / "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    # 分析错误案例
    if args.analyze_errors:
        analyze_errors(predictions, y_true, y_pred)

    return results


def analyze_errors(predictions, y_true, y_pred):
    """分析错误案例"""
    logger.info("\n***** Error Analysis *****")

    errors = []
    for i, (pred_sample, true_label, pred_label) in enumerate(
        zip(predictions, y_true, y_pred)
    ):
        if true_label != pred_label:
            errors.append(
                {
                    "idx": i,
                    "sample_id": pred_sample.get("sample_id"),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": pred_sample.get("confidence"),
                    "reasoning": pred_sample.get("reasoning", "")[:200],
                }
            )

    logger.info(f"Total errors: {len(errors)}")

    # 保存错误案例
    output_file = LEVEL2_OUTPUT / "task1" / "error_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    logger.info(f"Error analysis saved to {output_file}")

    # 打印前 5 个错误案例
    logger.info("\nSample errors:")
    for error in errors[:5]:
        logger.info(f"\n  Sample {error['idx']}:")
        logger.info(
            f"    True: {error['true_label']}, Pred: {error['pred_label']}, Conf: {error['confidence']:.2f}"
        )
        logger.info(f"    Reasoning: {error['reasoning']}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions_file",
        type=str,
        default="predictions.json",
        help="预测结果文件名",
    )
    parser.add_argument("--analyze_errors", action="store_true", help="分析错误案例")

    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
