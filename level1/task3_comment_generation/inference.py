import os
import torch
import logging
import argparse
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
import multiprocessing

from models import build_or_load_gen_model
from utils import CommentGenDataset, SimpleGenDataset
from configs import add_args, set_seed

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(args, model, tokenizer):
    pool = multiprocessing.Pool(args.cpu_count)

    # 加载测试数据
    if args.raw_input:
        eval_dataset = SimpleGenDataset(tokenizer, pool, args, args.test_filename)
    else:
        eval_dataset = CommentGenDataset(tokenizer, pool, args, args.test_filename)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=0,
                                 collate_fn=lambda x: x)

    logger.info(f"***** Running evaluation on {args.test_filename} *****")
    model.eval()
    pred_ids = []

    # 推理
    for examples in tqdm(eval_dataloader, desc="Inference"):
        examples = [ex for ex in examples if ex is not None]
        if len(examples) == 0: continue

        source_ids = torch.tensor([ex.source_ids for ex in examples], dtype=torch.long).to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_id)

        with torch.no_grad():
            preds = model.generate(source_ids,
                                   attention_mask=source_mask,
                                   use_cache=True,
                                   num_beams=args.beam_size,
                                   max_length=args.max_target_length,
                                   early_stopping=True)
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    # 解码
    # [1:] 是为了去除生成的起始 token (例如 <s> 或 <msg>)
    pred_nls = [tokenizer.decode(id[1:], skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in
                pred_ids]

    # 简单清洗
    pred_nls = [pred.strip() for pred in pred_nls]

    # 获取 Ground Truth
    golds = []
    with open(args.test_filename, "r") as f:
        for line in f:
            obj = json.loads(line)
            golds.append(obj.get("msg", "").strip())

    # 确保长度一致
    golds = golds[:len(pred_nls)]

    # 保存结果到文件
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    preds_file = os.path.join(args.output_dir, "predictions.txt")
    golds_file = os.path.join(args.output_dir, "references.txt")

    with open(preds_file, 'w', encoding="utf-8") as f:
        for pred in pred_nls:
            f.write(pred.replace("\n", " ") + '\n')

    with open(golds_file, 'w', encoding="utf-8") as f:
        for gold in golds:
            f.write(gold.replace("\n", " ") + '\n')

    logger.info(f"Predictions saved to {preds_file}")
    logger.info(f"References saved to {golds_file}")


def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.cpu_count = multiprocessing.cpu_count()

    set_seed(args)

    # 加载模型 (注意：推理时 load_model_path 指定 checkpoint 路径)
    # model_name_or_path 此时应指向 checkpoint
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(args.device)

    if args.do_test:
        evaluate(args, model, tokenizer)


if __name__ == "__main__":
    main()