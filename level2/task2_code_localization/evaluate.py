"""
Level 2 Task 2: 问题代码定位 - 评估脚本

评估问题定位的准确性
"""

import os
import sys
import argparse
import json
import logging
import numpy as np

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config import LEVEL2_OUTPUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_metrics(predictions):
    """计算评估指标"""
    exact_matches = 0
    partial_matches = 0
    reciprocal_ranks = []
    ious = []

    for pred in predictions:
        pred_lines = set(pred["prediction"])
        true_lines = set(pred["ground_truth"])

        if not true_lines:
            continue

        # Exact Match
        if pred_lines == true_lines:
            exact_matches += 1

        # Partial Match (至少一个正确)
        if pred_lines & true_lines:
            partial_matches += 1

        # IoU (Intersection over Union)
        if pred_lines or true_lines:
            intersection = len(pred_lines & true_lines)
            union = len(pred_lines | true_lines)
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)

        # MRR (Mean Reciprocal Rank)
        # 找到第一个正确预测的位置
        if pred["prediction"]:
            for rank, pred_line in enumerate(pred["prediction"], 1):
                if pred_line in true_lines:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(0.0)

    n = len(predictions)

    metrics = {
        "exact_match": exact_matches / n if n > 0 else 0.0,
        "partial_match": partial_matches / n if n > 0 else 0.0,
        "mean_iou": np.mean(ious) if ious else 0.0,
        "mrr": np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "total_samples": n,
    }

    # Top-K accuracy
    for k in [1, 3, 5]:
        top_k_correct = 0
        for pred in predictions:
            pred_lines = pred["prediction"][:k] if pred["prediction"] else []
            true_lines = set(pred["ground_truth"])

            if any(line in true_lines for line in pred_lines):
                top_k_correct += 1

        metrics[f"top_{k}_accuracy"] = top_k_correct / n if n > 0 else 0.0

    return metrics


def evaluate(args):
    """评估预测结果"""
    # 加载预测
    predictions_file = LEVEL2_OUTPUT / "task2" / args.predictions_file
    logger.info(f"Loading predictions from {predictions_file}")

    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    # 计算指标
    logger.info(f"Evaluating {len(predictions)} samples")
    metrics = calculate_metrics(predictions)

    # 打印结果
    logger.info("***** Evaluation Results *****")
    logger.info(f"  Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"  Partial Match: {metrics['partial_match']:.4f}")
    logger.info(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    logger.info(f"  MRR: {metrics['mrr']:.4f}")
    logger.info(f"  Top-1 Accuracy: {metrics['top_1_accuracy']:.4f}")
    logger.info(f"  Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    logger.info(f"  Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")

    # 保存结果
    output_file = LEVEL2_OUTPUT / "task2" / "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    # 错误分析
    if args.analyze_errors:
        analyze_errors(predictions)

    return metrics


def analyze_errors(predictions):
    """分析错误案例"""
    logger.info("\n***** Error Analysis *****")

    errors = []
    for pred in predictions:
        pred_lines = set(pred["prediction"])
        true_lines = set(pred["ground_truth"])

        if pred_lines != true_lines:
            errors.append(
                {
                    "idx": pred["idx"],
                    "sample_id": pred.get("sample_id"),
                    "predicted": pred["prediction"],
                    "ground_truth": pred["ground_truth"],
                    "missing": list(true_lines - pred_lines),
                    "extra": list(pred_lines - true_lines),
                    "confidence": pred.get("confidence", 0.0),
                    "reasoning": pred.get("reasoning", "")[:200],
                }
            )

    logger.info(f"Total errors: {len(errors)}")

    # 保存错误分析
    output_file = LEVEL2_OUTPUT / "task2" / "error_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    logger.info(f"Error analysis saved to {output_file}")

    # 打印示例
    logger.info("\nSample errors:")
    for error in errors[:5]:
        logger.info(f"\n  Sample {error['idx']}:")
        logger.info(f"    Predicted: {error['predicted']}")
        logger.info(f"    Ground Truth: {error['ground_truth']}")
        logger.info(f"    Missing: {error['missing']}")
        logger.info(f"    Extra: {error['extra']}")
        logger.info(f"    Confidence: {error['confidence']:.2f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions_file", type=str, default="predictions.json")
    parser.add_argument("--analyze_errors", action="store_true")

    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
