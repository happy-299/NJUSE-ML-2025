"""
Level 1 Task 1: 代码质量评估 - 测试脚本

在测试集上评估训练好的模型
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import RobertaTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
codereviewer_path = os.path.abspath("../../../CodeBERT-master/CodeReviewer/code")
sys.path.append(codereviewer_path)

from models import ClassificationModel
from train import load_data, set_seed

from config import DIFF_QUALITY_DIR, LEVEL1_CHECKPOINT_DIR, LEVEL1_OUTPUT, DEVICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test(args):
    """测试模型"""
    # 设置随机种子
    set_seed(args.seed)

    # 加载模型和 tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    model = ClassificationModel.from_pretrained(args.model_path)
    model.to(DEVICE)
    model.eval()

    # 加载测试数据
    test_dataset = load_data(
        DIFF_QUALITY_DIR, tokenizer, args.max_source_length, split="test"
    )

    # 创建 DataLoader
    sampler = SequentialSampler(test_dataset)
    dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=args.batch_size)

    # 推理
    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Batch size = {args.batch_size}")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
            labels = batch[2].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_macro": float(f1),
    }

    logger.info("***** Test Results *****")
    for key, value in results.items():
        logger.info(f"  {key} = {value:.4f}")

    # 保存结果
    output_dir = LEVEL1_OUTPUT / "task1"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "test_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default=str(LEVEL1_CHECKPOINT_DIR / "task1" / "checkpoint-best"),
        help="训练好的模型路径",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    test(args)


if __name__ == "__main__":
    main()
