"""
Level 1 Task 1: Code Quality Estimation - Test Script

Evaluate trained model on test set
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

# Add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
codereviewer_path = os.path.abspath("../../../CodeBERT-master/CodeReviewer/code")
sys.path.append(codereviewer_path)

from models import ReviewerModel
from train import set_seed, SimpleClsDataset

from config import DIFF_QUALITY_DIR, LEVEL1_CHECKPOINT_DIR, LEVEL1_OUTPUT, DEVICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(data_dir, tokenizer, max_length, sample_num=-1):
    """Load test dataset"""
    logger.info(f"Loading test data from {data_dir}")

    file_path = str(data_dir / "cls-test.jsonl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    # Use lightweight dataset
    dataset = SimpleClsDataset(
        file_paths=[file_path],
        tokenizer=tokenizer,
        max_length=max_length,
        samplenum=sample_num,
    )

    logger.info(f"Loaded {len(dataset)} test examples")
    return dataset


def test(args):
    """Test model"""
    # Set random seed
    set_seed(args.seed)

    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    model = ReviewerModel.from_pretrained(args.model_path)
    model.to(DEVICE)
    model.eval()

    # Load test data
    test_dataset = load_test_data(
        DIFF_QUALITY_DIR, tokenizer, args.max_source_length, args.sample_num
    )

    # Create DataLoader
    sampler = SequentialSampler(test_dataset)
    dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=args.batch_size)

    # Inference
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

            # Use cls method directly to get logits (without labels)
            logits = model.cls(
                input_ids=input_ids, attention_mask=attention_mask, labels=None
            )
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
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

    # Save results
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
        help="Trained model path",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample_num", type=int, default=-1, help="Test sample count, -1 for all"
    )

    args = parser.parse_args()

    test(args)


if __name__ == "__main__":
    main()
