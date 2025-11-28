"""
Level 1 Task 1: Code Quality Estimation - Training Script

Train a binary classification model based on CodeReviewer
Adapted from: https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Dataset
from torch.optim import AdamW
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import CodeReviewer modules
codereviewer_path = os.path.abspath("../../../CodeBERT-master/CodeReviewer/code")
sys.path.append(codereviewer_path)

from models import build_or_load_gen_model

# Import project config
from config import (
    DIFF_QUALITY_DIR,
    LEVEL1_CHECKPOINT_DIR,
    LEVEL1_OUTPUT,
    CODEREVIEWER_MODEL_NAME,
    LEVEL1_TRAIN_CONFIG,
    DEVICE,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class SimpleClsDataset(Dataset):
    """Simple classification dataset, memory efficient"""

    def __init__(self, file_paths, tokenizer, max_length=512, samplenum=-1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue

            logger.info(f"Reading {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Use 'patch' as input (contains diff info), 'y' as label
                        input_text = data.get("patch", "") or data.get("oldf", "")
                        label = int(data.get("y", data.get("label", 0)))
                        self.examples.append(
                            {
                                "input": input_text,
                                "label": label,
                            }
                        )
                    except json.JSONDecodeError:
                        continue

                    if samplenum > 0 and len(self.examples) >= samplenum:
                        break

            if samplenum > 0 and len(self.examples) >= samplenum:
                break

        logger.info(f"Loaded {len(self.examples)} examples")
        # Log label distribution to verify data correctness
        label_counts = {}
        for ex in self.examples:
            label_counts[ex["label"]] = label_counts.get(ex["label"], 0) + 1
        logger.info(f"Label distribution: {label_counts}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Encode input
        encoded = self.tokenizer(
            example["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return (
            encoded["input_ids"].squeeze(0),
            encoded["attention_mask"].squeeze(0),
            torch.tensor(example["label"], dtype=torch.long),
        )


def set_seed(seed=42):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_dir, tokenizer, args, split="train"):
    """Load dataset"""
    logger.info(f"Loading {split} data from {data_dir}")

    # Prepare file paths
    if split == "train":
        file_paths = []
        for i in range(4):
            file_path = data_dir / f"cls-train-chunk-{i}.jsonl"
            if file_path.exists():
                file_paths.append(str(file_path))
        if not file_paths:
            raise FileNotFoundError(f"No training data files found in {data_dir}")
    elif split == "valid":
        file_paths = [str(data_dir / "cls-valid.jsonl")]
    elif split == "test":
        file_paths = [str(data_dir / "cls-test.jsonl")]
    else:
        raise ValueError(f"Unknown split: {split}")

    # Use lightweight custom dataset
    # Determine sample count based on split
    if split == "train":
        samplenum = getattr(args, "sample_num", -1)
    elif split == "valid":
        samplenum = getattr(args, "valid_sample_num", -1)
    else:
        samplenum = getattr(args, "sample_num", -1)

    dataset = SimpleClsDataset(
        file_paths=file_paths,
        tokenizer=tokenizer,
        max_length=args.max_source_length,
        samplenum=samplenum,
    )

    return dataset


def train(args):
    """Train model"""
    # Set random seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and model using CodeReviewer's loader
    logger.info(f"Loading model from {args.model_name_or_path}")
    config, model, tokenizer = build_or_load_gen_model(args)

    model.to(DEVICE)

    # Load data
    train_dataset = load_data(DIFF_QUALITY_DIR, tokenizer, args, split="train")
    valid_dataset = load_data(DIFF_QUALITY_DIR, tokenizer, args, split="valid")

    # Create DataLoader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    # Optimizer and learning rate scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    num_training_steps = (
        len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {num_training_steps}")

    global_step = 0
    best_acc = 0.0
    model.zero_grad()

    for epoch in range(args.num_epochs):
        model.train()
        tr_loss = 0.0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, batch in enumerate(pbar):
            # Move data to device
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
            labels = batch[2].to(DEVICE)

            # Forward pass (use cls=True for classification)
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                cls=True,
            )

            # Backward pass
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                pbar.set_postfix({"loss": tr_loss / (step + 1)})

                # Periodic validation
                if global_step % args.save_steps == 0:
                    val_acc = evaluate(model, valid_dataset, args)
                    logger.info(f"Step {global_step}: Val Accuracy = {val_acc:.4f}")

                    # Save best model
                    if val_acc > best_acc:
                        best_acc = val_acc
                        output_dir = os.path.join(args.output_dir, f"checkpoint-best")
                        os.makedirs(output_dir, exist_ok=True)

                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        # Save training state for resume
                        torch.save(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "best_acc": best_acc,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                            },
                            os.path.join(output_dir, "training_state.pt"),
                        )
                        logger.info(f"Saved best model to {output_dir}")

                    model.train()

        # Validate after each epoch
        val_acc = evaluate(model, valid_dataset, args)
        logger.info(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.4f}")

        # Save checkpoint after each epoch (for resume)
        epoch_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_acc": best_acc,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(epoch_dir, "training_state.pt"),
        )
        logger.info(f"Saved epoch {epoch+1} checkpoint to {epoch_dir}")

    logger.info(f"Training completed. Best Val Accuracy = {best_acc:.4f}")


def evaluate(model, dataset, args):
    """Evaluate model"""
    model.eval()

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
            labels = batch[2].to(DEVICE)

            # Call cls method directly with labels=None to get logits
            logits = model.cls(
                input_ids=input_ids, attention_mask=attention_mask, labels=None
            )
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return accuracy


def main():
    parser = argparse.ArgumentParser()

    # Path arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=CODEREVIEWER_MODEL_NAME,
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Optional: Saved model path (for loading fine-tuned model)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(LEVEL1_CHECKPOINT_DIR / "task1"),
        help="Model output directory",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local process index (for distributed training)",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=LEVEL1_TRAIN_CONFIG["batch_size"]
    )
    parser.add_argument(
        "--learning_rate", type=float, default=LEVEL1_TRAIN_CONFIG["learning_rate"]
    )
    parser.add_argument(
        "--num_epochs", type=int, default=LEVEL1_TRAIN_CONFIG["num_epochs"]
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=LEVEL1_TRAIN_CONFIG["gradient_accumulation_steps"],
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=LEVEL1_TRAIN_CONFIG["warmup_steps"]
    )
    parser.add_argument(
        "--save_steps", type=int, default=LEVEL1_TRAIN_CONFIG["save_steps"]
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=LEVEL1_TRAIN_CONFIG["max_grad_norm"]
    )
    parser.add_argument("--seed", type=int, default=LEVEL1_TRAIN_CONFIG["seed"])

    # Model arguments
    parser.add_argument("--max_source_length", type=int, default=512)

    # Sampling arguments (for quick testing)
    parser.add_argument(
        "--sample_num",
        type=int,
        default=-1,
        help="Training sample count, -1 means use all data. Set small value (e.g., 5000) for faster training",
    )
    parser.add_argument(
        "--valid_sample_num",
        type=int,
        default=-1,
        help="Validation sample count, -1 means use all data. Set small value to speed up validation",
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
