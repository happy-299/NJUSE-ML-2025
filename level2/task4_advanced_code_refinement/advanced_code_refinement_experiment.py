"""
Level 2 Task 4: 高级代码修复生成实验

使用提示工程技术改进CodeReviewer模型性能
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import os
from datetime import datetime

class Level2CodeRefinement:
    """Level 2代码修复生成器 - 使用高级提示工程"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """加载CodeReviewer模型"""
        print("加载CodeReviewer模型...")
        model_name = "microsoft/codereviewer"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        print("✅ 模型加载完成")
        
    def create_advanced_prompt(self, comment, old_code, language="unknown"):
        """创建高级提示模板"""
        prompt = f"""Task: Fix the code based on the review comment.
Language: {language}
Review: {comment}
Original Code: {old_code}

Instructions:
- Generate ONLY the fixed code
- Do NOT generate review comments or explanations  
- Keep the same programming language and style
- Apply the suggested changes from the review

Fixed Code:"""
        
        return prompt
        
    def generate_code_fix(self, comment, old_code, language="unknown"):
        """生成代码修复"""
        try:
            prompt = self.create_advanced_prompt(comment, old_code, language)
            
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=400, 
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=200,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
            
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 后处理：提取"Fixed Code:"之后的内容
            if "Fixed Code:" in prediction:
                prediction = prediction.split("Fixed Code:")[-1].strip()
            
            return prediction
            
        except Exception as e:
            return f"[ERROR: {str(e)}]"

def load_test_data(file_path, max_samples=200):
    """加载测试数据"""
    print(f"加载数据: {os.path.basename(file_path)} (最多{max_samples}样本)")
    examples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                if all(key in data for key in ['old_hunk', 'comment', 'new']):
                    examples.append({
                        'old_hunk': data['old_hunk'],
                        'comment': data['comment'],
                        'target': data['new'],
                        'language': data.get('lang', 'unknown')
                    })
            except:
                continue
    
    print(f"成功加载 {len(examples)} 个样本")
    return examples

def run_level2_experiment():
    """运行Level 2实验"""
    print("="*70)
    print("Level 2 Task 4: 高级代码修复生成实验")  
    print("="*70)
    
    start_time = time.time()
    
    # 加载测试数据
    test_data = load_test_data('../../data/raw/ref-test.jsonl', max_samples=200)
    if not test_data:
        print("❌ 无法加载测试数据")
        return
    
    # 初始化Level 2系统
    level2_system = Level2CodeRefinement()
    level2_system.load_model()
    
    # 实验结果
    results = {
        'exact_matches': 0,
        'total_samples': len(test_data),
        'predictions': [],
        'language_stats': {}
    }
    
    print(f"\n开始Level 2实验 - 处理 {len(test_data)} 个样本...")
    
    for i, example in enumerate(test_data):
        if i % 50 == 0:
            print(f"  处理进度: {i+1}/{len(test_data)}")
        
        # Level 2: 高级提示工程生成
        prediction = level2_system.generate_code_fix(
            example['comment'],
            example['old_hunk'], 
            example['language']
        )
        
        # 评估
        target = example['target'].strip()
        pred_clean = prediction.strip()
        is_match = (pred_clean == target)
        
        if is_match:
            results['exact_matches'] += 1
        
        # 语言统计
        lang = example['language']
        if lang not in results['language_stats']:
            results['language_stats'][lang] = {'total': 0, 'correct': 0}
        results['language_stats'][lang]['total'] += 1
        if is_match:
            results['language_stats'][lang]['correct'] += 1
        
        # 保存详细结果（前10个样本）
        if i < 10:
            results['predictions'].append({
                'input_comment': example['comment'][:100] + "...",
                'input_code': example['old_hunk'][:200] + "...",
                'target': target[:200] + "..." if len(target) > 200 else target,
                'prediction': pred_clean[:200] + "..." if len(pred_clean) > 200 else pred_clean,
                'exact_match': is_match,
                'language': lang
            })
    
    # 计算最终指标
    accuracy = results['exact_matches'] / results['total_samples'] if results['total_samples'] > 0 else 0
    results['accuracy'] = accuracy
    
    # 显示结果
    print(f"\n📊 Level 2实验结果:")
    print(f"  总样本数: {results['total_samples']}")
    print(f"  精确匹配: {results['exact_matches']}")
    print(f"  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 与Level 1对比
    level1_accuracy = 0.0
    improvement = accuracy - level1_accuracy
    print(f"  相比Level 1提升: {improvement:.4f} ({improvement*100:+.2f}%)")
    
    # 各语言表现
    print(f"\n📈 各语言表现:")
    for lang, stats in sorted(results['language_stats'].items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] >= 5:
            lang_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {lang:<6}: {stats['correct']:>2}/{stats['total']:>2} ({lang_acc:.3f})")
    
    # 预测示例
    print(f"\n🔍 Level 2预测示例:")
    for i, result in enumerate(results['predictions'][:3]):
        print(f"\n  样例 {i+1} [{result['language']}]:")
        print(f"    评审: {result['input_comment']}")
        print(f"    目标: {result['target']}")
        print(f"    预测: {result['prediction']}")
        print(f"    匹配: {'✅' if result['exact_match'] else '❌'}")
    
    # 保存结果
    experiment_time = time.time() - start_time
    
    final_results = {
        'experiment_info': {
            'level': 'Level 2',
            'method': 'Advanced Prompt Engineering',
            'date': datetime.now().isoformat(),
            'duration_seconds': round(experiment_time, 2)
        },
        'results': results,
        'level2_techniques': {
            'prompt_engineering': {
                'structured_prompt': True,
                'task_clarification': True,
                'output_format_specification': True,
                'language_context': True
            },
            'generation_parameters': {
                'num_beams': 4,
                'temperature': 0.7,
                'do_sample': True,
                'repetition_penalty': 1.2
            }
        }
    }
    
    # 保存详细结果
    os.makedirs('../../outputs/level2/task4', exist_ok=True)
    result_path = '../../outputs/level2/task4/level2_experiment_results.json'
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n⏱️  实验耗时: {experiment_time:.1f} 秒")
    print(f"📄 详细结果已保存: {result_path}")
    print("\n" + "="*70)
    print("🎉 Level 2实验完成！")
    
    return final_results

if __name__ == "__main__":
    results = run_level2_experiment()