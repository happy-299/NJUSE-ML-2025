"""
Level 1 Task 4: 修复代码生成 - 推理脚本

使用训练好的模型对新输入进行代码修复生成
"""

import os
import sys
import argparse
import logging
import json
import torch
from transformers import RobertaTokenizer
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 导入项目配置
from config import LEVEL1_CHECKPOINT_DIR, DEVICE

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CodeRefinementInference:
    """代码修复推理类"""
    
    def __init__(self, model_path, max_source_length=512, max_target_length=128):
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # 加载模型和tokenizer
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        
        try:
            from transformers import RobertaForCausalLM
            self.model = RobertaForCausalLM.from_pretrained(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.model.to(DEVICE)
        self.model.eval()
    
    def preprocess_input(self, old_hunk, comment):
        """预处理输入"""
        # 构造输入格式: old_hunk + comment
        source = f"Old code: {old_hunk} <sep> Comment: {comment}"
        
        # Tokenize
        source_tokens = self.tokenizer.encode(
            source,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        source_mask = (source_tokens != self.tokenizer.pad_token_id).long()
        
        return source_tokens.to(DEVICE), source_mask.to(DEVICE)
    
    def generate_refined_code(self, old_hunk, comment, num_beams=5, temperature=1.0, do_sample=False):
        """生成修复后的代码"""
        # 预处理输入
        source_ids, source_mask = self.preprocess_input(old_hunk, comment)
        
        with torch.no_grad():
            # 生成
            generated_ids = self.model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                max_length=self.max_target_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # 解码
            # 移除输入部分,只保留生成部分
            input_len = source_ids.ne(self.tokenizer.pad_token_id).sum().item()
            if generated_ids.size(1) > input_len:
                generated_ids = generated_ids[:, input_len:]
            
            refined_code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
        return refined_code.strip()
    
    def batch_inference(self, input_data, num_beams=5, temperature=1.0, do_sample=False):
        """批量推理"""
        results = []
        
        for item in tqdm(input_data, desc="Generating refined code"):
            old_hunk = item.get('old_hunk', '')
            comment = item.get('comment', '')
            
            try:
                refined_code = self.generate_refined_code(
                    old_hunk, comment, num_beams, temperature, do_sample
                )
                
                result = {
                    'old_hunk': old_hunk,
                    'comment': comment,
                    'refined_code': refined_code,
                    'original_id': item.get('id', len(results)),
                    'language': item.get('lang', 'unknown')
                }
                
                if 'reference' in item:
                    result['reference'] = item['reference']
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process item {len(results)}: {e}")
                # 添加错误结果
                results.append({
                    'old_hunk': old_hunk,
                    'comment': comment,
                    'refined_code': f"ERROR: {str(e)}",
                    'original_id': item.get('id', len(results)),
                    'language': item.get('lang', 'unknown')
                })
        
        return results


def load_input_data(input_file):
    """加载输入数据"""
    logger.info(f"Loading input from {input_file}")
    
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            try:
                item = json.loads(line.strip())
                if 'old_hunk' in item and 'comment' in item:
                    item['id'] = line_idx
                    data.append(item)
                else:
                    logger.warning(f"Missing required fields in line {line_idx}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in line {line_idx}: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} input samples")
    return data


def save_results(results, output_file):
    """保存结果"""
    logger.info(f"Saving results to {output_file}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(results)} results")


def interactive_mode(inferencer):
    """交互模式"""
    logger.info("Starting interactive mode. Type 'quit' to exit.")
    
    while True:
        print("\n" + "="*50)
        print("Code Refinement Interactive Mode")
        print("="*50)
        
        # 获取输入
        print("\nEnter old code (multi-line, end with '###'):")
        old_hunk = ""
        while True:
            line = input()
            if line.strip() == '###':
                break
            old_hunk += line + "\n"
        
        if old_hunk.strip().lower() == 'quit':
            break
        
        comment = input("\nEnter comment: ").strip()
        if comment.lower() == 'quit':
            break
        
        # 生成修复代码
        try:
            refined_code = inferencer.generate_refined_code(old_hunk, comment)
            
            print("\n" + "-"*30)
            print("Refined Code:")
            print("-"*30)
            print(refined_code)
            print("-"*30)
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    
    # 路径参数
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(LEVEL1_CHECKPOINT_DIR / "task4" / "checkpoint-best"),
        help="训练好的模型路径"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="输入文件路径 (JSONL格式)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="输出文件路径"
    )
    
    # 模型参数
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    
    # 生成参数
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--do_sample", action='store_true')
    
    # 模式参数
    parser.add_argument("--interactive", action='store_true', help="启用交互模式")
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = CodeRefinementInference(
        args.model_path,
        args.max_source_length,
        args.max_target_length
    )
    
    if args.interactive:
        # 交互模式
        interactive_mode(inferencer)
    else:
        # 批量处理模式
        if not args.input_file or not args.output_file:
            logger.error("Input file and output file are required for batch mode")
            return
        
        # 加载输入数据
        input_data = load_input_data(args.input_file)
        
        if not input_data:
            logger.error("No valid input data found")
            return
        
        # 批量推理
        results = inferencer.batch_inference(
            input_data,
            args.num_beams,
            args.temperature,
            args.do_sample
        )
        
        # 保存结果
        save_results(results, args.output_file)
        
        # 显示示例
        logger.info("\nSample results:")
        for i, result in enumerate(results[:3]):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Comment: {result['comment'][:100]}...")
            logger.info(f"Refined: {result['refined_code'][:100]}...")


if __name__ == "__main__":
    main()