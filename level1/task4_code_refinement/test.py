"""
Level 1 Task 4: 修复代码生成 - 测试脚本

测试训练好的代码修复生成模型并计算评估指标
"""

import os
import sys
import argparse
import logging
import json
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import RobertaTokenizer
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 导入训练脚本中的数据集类
from train import CodeRefinementDataset, load_data

# 导入项目配置
from config import DATA_DIR, LEVEL1_CHECKPOINT_DIR, LEVEL1_OUTPUT, DEVICE

# 导入评估工具
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils")))
try:
    from metrics import calculate_bleu, calculate_rouge_l, calculate_exact_match
except ImportError:
    print("Warning: metrics module not found. Using simplified metrics.")
    
    def calculate_bleu(predictions, references):
        """简化版 BLEU 计算"""
        from nltk.translate.bleu_score import corpus_bleu
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        refs = [[ref.split()] for ref in references]
        preds = [pred.split() for pred in predictions]
        return corpus_bleu(refs, preds)
    
    def calculate_rouge_l(predictions, references):
        """简化版 ROUGE-L 计算"""
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores(predictions, references, avg=True)
        return scores['rouge-l']['f']
    
    def calculate_exact_match(predictions, references):
        """计算完全匹配率"""
        matches = [pred.strip() == ref.strip() for pred, ref in zip(predictions, references)]
        return np.mean(matches)

try:
    # 尝试导入 CodeBLEU
    from codebleu import calc_codebleu
except ImportError:
    print("Warning: CodeBLEU not available. Please install: pip install codebleu")
    def calc_codebleu(references, predictions, lang):
        return {"codebleu": 0.0}

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_model(model_path):
    """加载训练好的模型"""
    logger.info(f"Loading model from {model_path}")
    
    # 加载 tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    
    # 加载模型
    try:
        from transformers import RobertaForCausalLM
        model = RobertaForCausalLM.from_pretrained(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    model.to(DEVICE)
    model.eval()
    
    return model, tokenizer


def generate_predictions(model, tokenizer, dataset, args):
    """生成预测结果"""
    model.eval()
    
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
            source_ids = batch['source_ids'].to(DEVICE)
            source_mask = batch['source_mask'].to(DEVICE)
            
            # 生成文本
            generated_ids = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                max_length=args.max_target_length,
                num_beams=args.num_beams,
                temperature=args.temperature,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # 解码生成的文本
            for i, generated in enumerate(generated_ids):
                # 移除输入部分,只保留生成部分
                input_len = source_ids[i].ne(tokenizer.pad_token_id).sum().item()
                if len(generated) > input_len:
                    generated = generated[input_len:]
                
                pred_text = tokenizer.decode(generated, skip_special_tokens=True)
                predictions.append(pred_text.strip())
                
                # 获取真实标签
                target_ids = batch['target_ids'][i]
                target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
                references.append(target_text.strip())
            
            # 限制样本数量进行快速测试
            if args.max_test_samples and len(predictions) >= args.max_test_samples:
                predictions = predictions[:args.max_test_samples]
                references = references[:args.max_test_samples]
                break
    
    return predictions, references


def calculate_metrics(predictions, references, languages=None):
    """计算各种评估指标"""
    metrics = {}
    
    # 1. Exact Match
    em_score = calculate_exact_match(predictions, references)
    metrics['exact_match'] = em_score
    
    # 2. BLEU-4
    try:
        bleu_score = calculate_bleu(predictions, references)
        metrics['bleu4'] = bleu_score
    except Exception as e:
        logger.warning(f"BLEU calculation failed: {e}")
        metrics['bleu4'] = 0.0
    
    # 3. ROUGE-L
    try:
        rouge_score = calculate_rouge_l(predictions, references)
        metrics['rouge_l'] = rouge_score
    except Exception as e:
        logger.warning(f"ROUGE-L calculation failed: {e}")
        metrics['rouge_l'] = 0.0
    
    # 4. CodeBLEU (如果可用)
    try:
        if languages:
            # 按语言分组计算 CodeBLEU
            lang_groups = defaultdict(list)
            for pred, ref, lang in zip(predictions, references, languages):
                lang_groups[lang].append((pred, ref))
            
            codebleu_scores = []
            for lang, items in lang_groups.items():
                if len(items) < 5:  # 跳过样本太少的语言
                    continue
                    
                lang_preds = [item[0] for item in items]
                lang_refs = [item[1] for item in items]
                
                try:
                    codebleu_result = calc_codebleu(lang_refs, lang_preds, lang=lang)
                    codebleu_scores.append(codebleu_result.get('codebleu', 0.0))
                except:
                    continue
            
            if codebleu_scores:
                metrics['codebleu'] = np.mean(codebleu_scores)
            else:
                metrics['codebleu'] = 0.0
        else:
            # 使用默认语言计算
            codebleu_result = calc_codebleu(references, predictions, lang='java')
            metrics['codebleu'] = codebleu_result.get('codebleu', 0.0)
    except Exception as e:
        logger.warning(f"CodeBLEU calculation failed: {e}")
        metrics['codebleu'] = 0.0
    
    return metrics


def test(args):
    """测试模型"""
    # 加载模型
    model, tokenizer = load_model(args.model_path)
    
    # 加载测试数据
    test_dataset = load_data(
        DATA_DIR / "raw" / "ref-test.jsonl",
        tokenizer,
        args.max_source_length,
        args.max_target_length,
        max_samples=args.max_test_samples
    )
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # 生成预测
    predictions, references = generate_predictions(model, tokenizer, test_dataset, args)
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    # 获取语言信息（如果需要）
    languages = None
    if hasattr(test_dataset, 'examples'):
        languages = [ex['lang'] for ex in test_dataset.examples[:len(predictions)]]
    
    # 计算指标
    metrics = calculate_metrics(predictions, references, languages)
    
    # 输出结果
    logger.info("\n" + "="*50)
    logger.info("Test Results:")
    logger.info("="*50)
    for metric, score in metrics.items():
        logger.info(f"{metric.upper()}: {score:.4f}")
    
    # 保存结果
    output_dir = LEVEL1_OUTPUT / "task4"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存预测结果
    results_file = output_dir / "predictions.jsonl"
    with open(results_file, 'w', encoding='utf-8') as f:
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            result = {
                'id': i,
                'prediction': pred,
                'reference': ref,
                'language': languages[i] if languages else 'unknown'
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"Predictions saved to {results_file}")
    
    # 保存评估指标
    metrics_file = output_dir / "test_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Metrics saved to {metrics_file}")
    
    # 显示一些示例
    logger.info("\nSample Predictions:")
    logger.info("-" * 50)
    for i in range(min(3, len(predictions))):
        logger.info(f"Example {i+1}:")
        logger.info(f"Reference: {references[i][:100]}{'...' if len(references[i]) > 100 else ''}")
        logger.info(f"Prediction: {predictions[i][:100]}{'...' if len(predictions[i]) > 100 else ''}")
        logger.info("-" * 30)
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    
    # 路径参数
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(LEVEL1_CHECKPOINT_DIR / "task4" / "checkpoint-best"),
        help="训练好的模型路径"
    )
    
    # 测试参数
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--max_test_samples", type=int, default=1000,
                       help="测试样本数量 (用于快速评估)")
    
    # 生成参数
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", type=bool, default=False)
    
    args = parser.parse_args()
    
    test(args)


if __name__ == "__main__":
    main()