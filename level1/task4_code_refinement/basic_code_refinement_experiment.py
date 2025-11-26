"""
Level 1 Task 4: 代码修复生成 - 真实数据实验

使用CodeReviewer模型在真实数据上进行完整实验
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import os
from datetime import datetime

def load_data(file_path, max_samples=None):
    """加载真实数据"""
    if max_samples:
        print(f"加载数据: {os.path.basename(file_path)} (最多{max_samples}样本)")
    else:
        print(f"加载数据: {os.path.basename(file_path)} (全部数据)")
    examples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_samples and idx >= max_samples:
                break
            try:
                data = json.loads(line.strip())
                if all(key in data for key in ['old_hunk', 'comment', 'new']):
                    examples.append({
                        'input': f"Review: {data['comment']} Code: {data['old_hunk']}",
                        'target': data['new'],
                        'lang': data.get('lang', 'unknown')
                    })
            except:
                continue
    
    print(f"成功加载 {len(examples)} 个样本")
    return examples

def run_experiment():
    """运行完整实验"""
    print("="*70)
    print("Level 1 Task 4: 代码修复生成实验")
    print("="*70)
    
    start_time = time.time()
    
    # 1. 加载数据
    print("\n[1/4] 数据加载")
    test_data = load_data('../../data/raw/ref-test.jsonl')  # 使用全部测试数据
    
    if not test_data:
        print("❌ 无法加载数据")
        return
    
    # 2. 加载模型
    print("\\n[2/4] 模型加载")
    try:
        print("  加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")
        print("  加载模型...")
        model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/codereviewer")
        model.eval()
        print("  ✅ 模型加载成功")
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return
    
    # 3. 运行推理
    print("\\n[3/4] 模型推理")
    results = []
    exact_matches = 0
    
    for i, example in enumerate(test_data):
        if i % 10 == 0:
            print(f"  处理进度: {i+1}/{len(test_data)}")
        
        try:
            # 生成预测
            inputs = tokenizer.encode(
                example['input'], 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=256,
                    num_beams=2,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            target = example['target'].strip()
            
            # 计算精确匹配
            is_match = prediction == target
            if is_match:
                exact_matches += 1
            
            results.append({
                'input': example['input'][:100] + "...",
                'target': target,
                'prediction': prediction,
                'exact_match': is_match,
                'language': example['lang']
            })
            
        except Exception as e:
            print(f"  样本{i}处理失败: {e}")
            results.append({
                'input': example['input'][:100] + "...",
                'target': example['target'],
                'prediction': "[ERROR]",
                'exact_match': False,
                'language': example['lang']
            })
    
    # 4. 分析结果
    print("\\n[4/4] 结果分析")
    total_samples = len(results)
    accuracy = exact_matches / total_samples if total_samples > 0 else 0
    
    # 按语言统计
    lang_stats = {}
    for result in results:
        lang = result['language']
        if lang not in lang_stats:
            lang_stats[lang] = {'total': 0, 'correct': 0}
        lang_stats[lang]['total'] += 1
        if result['exact_match']:
            lang_stats[lang]['correct'] += 1
    
    # 显示结果
    print(f"\\n📊 实验结果:")
    print(f"  总样本数: {total_samples}")
    print(f"  精确匹配: {exact_matches}")
    print(f"  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\\n📈 各语言表现:")
    for lang, stats in sorted(lang_stats.items(), key=lambda x: x[1]['total'], reverse=True):
        if stats['total'] >= 3:  # 只显示样本数>=3的语言
            lang_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {lang:<6}: {stats['correct']:>2}/{stats['total']:>2} ({lang_acc:.3f})")
    
    # 展示预测样例
    print(f"\\n🔍 预测示例:")
    for i, result in enumerate(results[:3]):
        print(f"\\n  样例 {i+1} [{result['language']}]:")
        print(f"    输入: {result['input']}")
        print(f"    目标: {result['target']}")
        print(f"    预测: {result['prediction']}")
        print(f"    匹配: {'✅' if result['exact_match'] else '❌'}")
    
    # 保存结果
    experiment_time = time.time() - start_time
    
    report = {
        'experiment_info': {
            'date': datetime.now().isoformat(),
            'duration_seconds': round(experiment_time, 2),
            'model': 'microsoft/codereviewer'
        },
        'results': {
            'total_samples': total_samples,
            'exact_matches': exact_matches,
            'accuracy': accuracy,
            'language_stats': lang_stats
        },
        'sample_predictions': results[:10]
    }
    
    # 确保输出目录存在
    os.makedirs('../../outputs/level1/task4', exist_ok=True)
    
    # 保存详细报告
    report_path = '../../outputs/level1/task4/experiment_results.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\\n⏱️  实验耗时: {experiment_time:.1f} 秒")
    print(f"📄 详细结果已保存: {report_path}")
    print("\\n" + "="*70)
    print("🎉 实验完成！")
    
    return report

if __name__ == "__main__":
    report = run_experiment()