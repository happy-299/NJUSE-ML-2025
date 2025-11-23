"""
Level 1 Task 4: 修复代码生成 - 训练脚本

基于 CodeReviewer 模型训练代码修复生成模型
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

# 导入 CodeReviewer 相关模块 (需要下载 CodeBERT 仓库)
try:
    codereviewer_path = os.path.abspath("../../../CodeBERT-master/CodeReviewer/code")
    sys.path.append(codereviewer_path)
    from models import build_or_load_gen_model
except ImportError:
    print("Warning: CodeBERT CodeReviewer not found. Please download from:")
    print("https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer")
    # 使用简化的模型加载
    from transformers import RobertaForSequenceClassification as build_or_load_gen_model

# 导入项目配置
from config import (
    DATA_DIR,
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


class CodeRefinementDataset(torch.utils.data.Dataset):
    """代码修复数据集"""
    
    def __init__(self, examples, tokenizer, max_source_length=512, max_target_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 构造输入: old_hunk + comment
        source = f"Old code: {example['old_hunk']} <sep> Comment: {example['comment']}"
        target = example['new']
        
        # Tokenize source
        source_tokens = self.tokenizer.encode(
            source,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).squeeze()
        
        # Tokenize target
        target_tokens = self.tokenizer.encode(
            target,
            max_length=self.max_target_length,
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        ).squeeze()
        
        # Create attention masks
        source_mask = (source_tokens != self.tokenizer.pad_token_id).long()
        target_mask = (target_tokens != self.tokenizer.pad_token_id).long()
        
        return {
            'source_ids': source_tokens,
            'source_mask': source_mask,
            'target_ids': target_tokens,
            'target_mask': target_mask
        }


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_file, tokenizer, max_source_length, max_target_length, max_samples=None):
    """加载数据集"""
    logger.info(f"Loading data from {data_file}")
    
    examples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
                
            try:
                data = json.loads(line.strip())
                
                # 验证必要字段
                if not all(key in data for key in ['old_hunk', 'comment', 'new']):
                    continue
                    
                # 过滤过长的样本
                if len(data['new']) > max_target_length * 4:  # 粗略估计token长度
                    continue
                    
                examples.append({
                    'old_hunk': data['old_hunk'],
                    'comment': data['comment'], 
                    'new': data['new'],
                    'lang': data.get('lang', 'unknown')
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skip invalid line {idx}: {e}")
                continue
    
    logger.info(f"Loaded {len(examples)} examples")
    
    # 创建数据集
    dataset = CodeRefinementDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length
    )
    
    return dataset


def train(args):
    """训练模型"""
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    
    # 添加特殊token
    special_tokens = {'sep_token': '<sep>'}
    tokenizer.add_special_tokens(special_tokens)
    
    # 加载模型
    logger.info(f"Loading model from {args.model_name_or_path}")
    try:
        # 尝试加载 CodeReviewer 模型
        config, model, _ = build_or_load_gen_model(args)
        model.resize_token_embeddings(len(tokenizer))
    except:
        # 后备方案：使用标准 Roberta 模型
        from transformers import RobertaForCausalLM
        model = RobertaForCausalLM.from_pretrained(args.model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))
    
    model.to(DEVICE)
    
    # 加载数据
    train_dataset = load_data(
        DATA_DIR / "raw" / "ref-train.jsonl",
        tokenizer,
        args.max_source_length,
        args.max_target_length,
        max_samples=args.max_train_samples
    )
    
    valid_dataset = load_data(
        DATA_DIR / "raw" / "ref-valid.jsonl", 
        tokenizer,
        args.max_source_length,
        args.max_target_length,
        max_samples=args.max_valid_samples
    )
    
    # 创建 DataLoader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size
    )
    
    # 优化器和学习率调度
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
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
    best_loss = float('inf')
    model.zero_grad()
    
    for epoch in range(args.num_epochs):
        model.train()
        tr_loss = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, batch in enumerate(pbar):
            # 将数据移到设备
            source_ids = batch['source_ids'].to(DEVICE)
            target_ids = batch['target_ids'].to(DEVICE)
            source_mask = batch['source_mask'].to(DEVICE)
            
            # 前向传播
            outputs = model(
                input_ids=source_ids,
                attention_mask=source_mask,
                labels=target_ids
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
                
                pbar.set_postfix({'loss': tr_loss / (step + 1)})
                
                # 定期保存和验证
                if global_step % args.save_steps == 0:
                    val_loss = evaluate(model, valid_dataset, tokenizer, args)
                    logger.info(f"Step {global_step}: Val Loss = {val_loss:.4f}")
                    
                    # 保存最佳模型
                    if val_loss < best_loss:
                        best_loss = val_loss
                        output_dir = os.path.join(args.output_dir, "checkpoint-best")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        logger.info(f"Saved best model to {output_dir}")
                    
                    model.train()
        
        # 每个 epoch 后验证
        val_loss = evaluate(model, valid_dataset, tokenizer, args)
        logger.info(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
    
    logger.info(f"Training completed. Best Val Loss = {best_loss:.4f}")


def evaluate(model, dataset, tokenizer, args):
    """评估模型"""
    model.eval()
    
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            source_ids = batch['source_ids'].to(DEVICE)
            target_ids = batch['target_ids'].to(DEVICE)
            source_mask = batch['source_mask'].to(DEVICE)
            
            outputs = model(
                input_ids=source_ids,
                attention_mask=source_mask,
                labels=target_ids
            )
            
            total_loss += outputs.loss.item()
            total_steps += 1
    
    return total_loss / total_steps


def main():
    parser = argparse.ArgumentParser()
    
    # 路径参数
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=CODEREVIEWER_MODEL_NAME,
        help="预训练模型名称或路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(LEVEL1_CHECKPOINT_DIR / "task4"),
        help="模型输出目录"
    )
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    # 模型参数
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    
    # 数据采样参数
    parser.add_argument("--max_train_samples", type=int, default=50000, 
                       help="训练集最大样本数 (用于快速实验)")
    parser.add_argument("--max_valid_samples", type=int, default=5000,
                       help="验证集最大样本数")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()