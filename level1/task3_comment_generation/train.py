import os
import torch
import logging
import argparse
import random
import json
import multiprocessing
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# 引入本地模块
from models import build_or_load_gen_model
from configs import add_args, set_seed
from utils import CommentGenDataset, SimpleGenDataset
# 确保 evaluator 文件夹存在
from evaluator.smooth_bleu import bleu_fromstr

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_loader(data_file, args, tokenizer, pool, eval=False):
    def fn(features):
        return features

    # 根据参数选择数据集格式
    if args.raw_input:
        dataset = SimpleGenDataset(tokenizer, pool, args, data_file)
    else:
        dataset = CommentGenDataset(tokenizer, pool, args, data_file)

    logger.info(f"Data length: {len(dataset)}.")

    if eval:
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset)  # 单卡使用 RandomSampler

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size,
                            num_workers=0, collate_fn=fn)
    return dataset, sampler, dataloader


def eval_bleu_epoch(args, eval_dataloader, model, tokenizer):
    logger.info(f"  ***** Running bleu evaluation on {args.eval_file} *****")
    model.eval()

    pred_ids, ex_ids = [], []
    # 使用 tqdm 显示进度
    for step, examples in enumerate(tqdm(eval_dataloader, desc="Eval")):
        source_ids = torch.tensor(
            [ex.source_ids for ex in examples], dtype=torch.long
        ).to(args.device)
        ids = [ex.example_id for ex in examples]
        source_mask = source_ids.ne(tokenizer.pad_id)

        with torch.no_grad():
            preds = model.generate(source_ids,
                                   attention_mask=source_mask,
                                   use_cache=True,
                                   num_beams=args.beam_size,
                                   early_stopping=True,
                                   max_length=args.max_target_length)
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    # 解码 [1:] 去除开头的 <msg>
    pred_nls = [tokenizer.decode(id[1:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                pred_ids]

    # 后处理
    for i in range(len(pred_nls)):
        chars = "(_)`."
        for c in chars:
            pred_nls[i] = pred_nls[i].replace(c, " " + c + " ")
        pred_nls[i] = " ".join(pred_nls[i].split())

    valid_file = args.dev_filename
    golds = []
    with open(valid_file, "r") as f:
        for line in f:
            obj = json.loads(line)
            if "msg" in obj:
                golds.append(obj["msg"])
            else:
                golds.append("")  # 防止没有msg字段报错

    golds = golds[:len(pred_nls)]

    # 简单的 BLEU 计算用于选择 Checkpoint
    bleu = bleu_fromstr(pred_nls, golds, rmstop=False)
    return bleu


def save_model(model, optimizer, scheduler, output_dir, config):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    config.save_pretrained(output_dir)

    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    logger.info(f"Saved model to {output_dir}")


def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    logger.info(f"Device: {device}, n_gpu: {args.n_gpu}")

    set_seed(args)

    # 加载模型
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)

    pool = multiprocessing.Pool(args.cpu_count)

    # 优化器配置
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # 计算总步数
    # 注意：这里需要先加载数据才能知道 dataset 大小，为了简化逻辑，我们先加载 train_loader
    train_file = args.train_filename
    _, _, train_dataloader = get_loader(train_file, args, tokenizer, pool)

    if args.train_steps == -1:
        # 如果未指定 train_steps，则按 epochs 计算
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.train_epochs
        args.train_steps = t_total

    args.warmup_steps = int(args.train_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.train_steps)

    logger.info(f"Total training steps: {args.train_steps}")

    global_step = 0
    tr_loss = 0.0
    best_bleu = 0.0

    model.zero_grad()

    for epoch in range(1, args.train_epochs + 1):
        logger.info(f"Starting Epoch {epoch}")
        model.train()

        for step, examples in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            if len(examples) == 0: continue

            source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(device)
            target_ids = torch.tensor([ex.target_ids for ex in examples], dtype=torch.long).to(device)

            source_mask = source_ids.ne(tokenizer.pad_id)
            target_mask = target_ids.ne(tokenizer.pad_id)

            loss = model(
                input_ids=source_ids,
                input_labels=None,
                decoder_input_ids=target_ids,
                attention_mask=source_mask,
                decoder_attention_mask=target_mask,
                encoder_loss=False
            )

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

                if args.log_steps > 0 and global_step % args.log_steps == 0:
                    logger.info(f"Step {global_step} | Loss: {tr_loss / global_step:.4f}")

                # 保存/评估逻辑
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # 验证
                    valid_file = args.dev_filename
                    _, _, valid_dataloader = get_loader(valid_file, args, tokenizer, pool, eval=True)
                    bleu = eval_bleu_epoch(args, valid_dataloader, model, tokenizer)
                    logger.info(f"Step {global_step} | BLEU: {bleu}")

                    output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_model(model, optimizer, scheduler, output_dir, config)

            if global_step >= args.train_steps:
                break

        if global_step >= args.train_steps:
            break

    # 训练结束，保存最终模型
    logger.info("Training finished. Saving last model.")
    output_dir = os.path.join(args.output_dir, "checkpoint-last")
    save_model(model, optimizer, scheduler, output_dir, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.cpu_count = multiprocessing.cpu_count()
    # 忽略长 tokenization 警告
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    main(args)