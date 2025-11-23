"""
Level 1 Task 1: 代码质量评估 - 训练脚本

基于 CodeReviewer 模型训练代码质量评估(二分类)模型
改编自: https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.optim import AdamW
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import json

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 导入 CodeReviewer 相关模块
codereviewer_path = os.path.abspath("../../../CodeBERT-master/CodeReviewer/code")
sys.path.append(codereviewer_path)

from models import build_or_load_gen_model
from utils import CommentClsDataset, ReviewExample

# 导入项目配置
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


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_dir, tokenizer, max_source_length, split="train"):
    """加载数据集"""
    logger.info(f"Loading {split} data from {data_dir}")

    examples = []

    if split == "train":
        # 训练集有多个文件
        data_files = [
            data_dir / "cls-train-chunk-0.jsonl",
            data_dir / "cls-train-chunk-1.jsonl",
            data_dir / "cls-train-chunk-2.jsonl",
            data_dir / "cls-train-chunk-3.jsonl",
        ]
    elif split == "valid":
        data_files = [data_dir / "cls-valid.jsonl"]
    elif split == "test":
        data_files = [data_dir / "cls-test.jsonl"]
    else:
        raise ValueError(f"Unknown split: {split}")

    for data_file in data_files:
        if not data_file.exists():
            logger.warning(f"Data file not found: {data_file}")
            continue

        with open(data_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line.strip())

                # 构造 ReviewExample
                example = ReviewExample(
                    idx=f"{data_file.stem}_{idx}",
                    old_code=data.get("oldf", ""),
                    old_comment="",
                    diff_code=data.get("old_hunk", ""),
                    new_code="",
                    new_comment="",
                    label=data.get("label", 0),
                )
                examples.append(example)

    logger.info(f"Loaded {len(examples)} examples from {split} set")

    # 创建数据集
    dataset = CommentClsDataset(
        examples=examples,
        tokenizer=tokenizer,
        args=argparse.Namespace(max_source_length=max_source_length),
    )

    return dataset


def train(args):
    """训练模型"""
    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizer 和模型 (使用 CodeReviewer 提供的加载函数)
    logger.info(f"Loading model from {args.model_name_or_path}")
    # models.build_or_load_gen_model 会返回 (config, model, tokenizer)
    config, model, tokenizer = build_or_load_gen_model(args)

    model.to(DEVICE)

    # 加载数据
    train_dataset = load_data(
        DIFF_QUALITY_DIR, tokenizer, args.max_source_length, split="train"
    )
    valid_dataset = load_data(
        DIFF_QUALITY_DIR, tokenizer, args.max_source_length, split="valid"
    )

    # 创建 DataLoader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    # 优化器和学习率调度
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

    # 训练循环
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
            # 将数据移到设备
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
            labels = batch[2].to(DEVICE)

            # 前向传播
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # 反向传播
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

                # 定期验证
                if global_step % args.save_steps == 0:
                    val_acc = evaluate(model, valid_dataset, args)
                    logger.info(f"Step {global_step}: Val Accuracy = {val_acc:.4f}")

                    # 保存最佳模型
                    if val_acc > best_acc:
                        best_acc = val_acc
                        output_dir = os.path.join(args.output_dir, f"checkpoint-best")
                        os.makedirs(output_dir, exist_ok=True)

                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        logger.info(f"Saved best model to {output_dir}")

                    model.train()

        # 每个 epoch 后验证
        val_acc = evaluate(model, valid_dataset, args)
        logger.info(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.4f}")

    logger.info(f"Training completed. Best Val Accuracy = {best_acc:.4f}")


def evaluate(model, dataset, args):
    """评估模型"""
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return accuracy


def main():
    parser = argparse.ArgumentParser()

    # 路径参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=CODEREVIEWER_MODEL_NAME,
        help="预训练模型名称或路径",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="可选: 已保存的模型路径 (用于加载 fine-tuned 模型)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(LEVEL1_CHECKPOINT_DIR / "task1"),
        help="模型输出目录",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="本地进程索引 (用于分布式训练)"
    )

    # 训练参数
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

    # 模型参数
    parser.add_argument("--max_source_length", type=int, default=512)

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
