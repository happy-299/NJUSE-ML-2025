"""
Level 1 Task 4: 代码修复生成 - 数据预处理脚本

对原始代码修复数据进行预处理和统计分析
"""

import os
import sys
import json
import argparse
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config import DATA_DIR

def load_and_analyze_data(data_file):
    """加载并分析数据"""
    print(f"Analyzing {data_file}...")
    
    examples = []
    language_counts = Counter()
    code_lengths = []
    comment_lengths = []
    refined_lengths = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(tqdm(f, desc="Processing")):
            try:
                data = json.loads(line.strip())
                
                # 验证必要字段
                if not all(key in data for key in ['old_hunk', 'comment', 'new']):
                    continue
                
                # 统计语言分布
                lang = data.get('lang', 'unknown')
                language_counts[lang] += 1
                
                # 统计长度分布
                code_lengths.append(len(data['old_hunk']))
                comment_lengths.append(len(data['comment']))
                refined_lengths.append(len(data['new']))
                
                examples.append({
                    'old_hunk': data['old_hunk'],
                    'comment': data['comment'],
                    'new': data['new'],
                    'lang': lang,
                    'hunk': data.get('hunk', ''),
                    'old': data.get('old', ''),
                    'repo': data.get('repo', 'unknown')
                })
                
            except (json.JSONDecodeError, KeyError) as e:
                continue
    
    print(f"\nLoaded {len(examples)} valid examples")
    
    # 统计信息
    stats = {
        'total_samples': len(examples),
        'language_distribution': dict(language_counts.most_common()),
        'length_stats': {
            'old_hunk': {
                'mean': np.mean(code_lengths),
                'std': np.std(code_lengths),
                'max': np.max(code_lengths),
                'min': np.min(code_lengths)
            },
            'comment': {
                'mean': np.mean(comment_lengths),
                'std': np.std(comment_lengths),
                'max': np.max(comment_lengths),
                'min': np.min(comment_lengths)
            },
            'refined_code': {
                'mean': np.mean(refined_lengths),
                'std': np.std(refined_lengths), 
                'max': np.max(refined_lengths),
                'min': np.min(refined_lengths)
            }
        }
    }
    
    return examples, stats

def print_statistics(stats, split_name):
    """打印统计信息"""
    print(f"\n{'='*50}")
    print(f"{split_name} Statistics")
    print(f"{'='*50}")
    
    print(f"Total samples: {stats['total_samples']}")
    
    print("\nLanguage distribution:")
    for lang, count in list(stats['language_distribution'].items())[:10]:
        percentage = count / stats['total_samples'] * 100
        print(f"  {lang}: {count} ({percentage:.1f}%)")
    
    print("\nLength statistics:")
    for field, length_stats in stats['length_stats'].items():
        print(f"  {field}:")
        print(f"    Mean: {length_stats['mean']:.1f}")
        print(f"    Std: {length_stats['std']:.1f}")
        print(f"    Range: {length_stats['min']} - {length_stats['max']}")

def create_sample_dataset(examples, output_file, sample_size, strategy='random'):
    """创建采样数据集"""
    print(f"\nCreating sample dataset: {sample_size} samples")
    
    if strategy == 'random':
        # 随机采样
        import random
        sampled = random.sample(examples, min(sample_size, len(examples)))
    elif strategy == 'balanced':
        # 按语言均衡采样
        lang_groups = defaultdict(list)
        for ex in examples:
            lang_groups[ex['lang']].append(ex)
        
        # 每种语言采样相同数量
        samples_per_lang = sample_size // len(lang_groups)
        sampled = []
        
        for lang, lang_examples in lang_groups.items():
            import random
            n_samples = min(samples_per_lang, len(lang_examples))
            sampled.extend(random.sample(lang_examples, n_samples))
        
        # 如果还需要更多样本，随机补充
        if len(sampled) < sample_size:
            remaining = sample_size - len(sampled)
            remaining_examples = [ex for ex in examples if ex not in sampled]
            if remaining_examples:
                sampled.extend(random.sample(remaining_examples, 
                                           min(remaining, len(remaining_examples))))
    
    # 保存采样数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in sampled:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(sampled)} samples to {output_file}")
    return sampled

def visualize_statistics(train_stats, valid_stats, test_stats, output_dir):
    """可视化统计信息"""
    print("\nCreating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 语言分布对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (stats, name) in enumerate([(train_stats, 'Train'), 
                                       (valid_stats, 'Valid'), 
                                       (test_stats, 'Test')]):
        langs = list(stats['language_distribution'].keys())[:8]
        counts = [stats['language_distribution'][lang] for lang in langs]
        
        axes[i].bar(langs, counts)
        axes[i].set_title(f'{name} Language Distribution')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'language_distribution.png'))
    plt.close()
    
    # 2. 长度分布对比
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    fields = ['old_hunk', 'comment', 'refined_code']
    datasets = [(train_stats, 'Train'), (valid_stats, 'Valid'), (test_stats, 'Test')]
    
    for i, field in enumerate(fields):
        for j, (stats, name) in enumerate(datasets):
            length_stat = stats['length_stats'][field]
            
            # 创建模拟分布用于可视化
            mean, std = length_stat['mean'], length_stat['std']
            x = np.linspace(max(0, mean - 3*std), mean + 3*std, 100)
            y = np.exp(-0.5 * ((x - mean) / std) ** 2)
            
            axes[i][j].plot(x, y)
            axes[i][j].axvline(mean, color='red', linestyle='--', alpha=0.7)
            axes[i][j].set_title(f'{name} {field} Length')
            axes[i][j].set_xlabel('Length')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_distributions.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def show_examples(examples, n=5):
    """显示数据示例"""
    print(f"\n{'='*60}")
    print(f"Sample Examples ({n} samples)")
    print(f"{'='*60}")
    
    for i, example in enumerate(examples[:n]):
        print(f"\n--- Example {i+1} ---")
        print(f"Language: {example['lang']}")
        print(f"Repository: {example.get('repo', 'unknown')}")
        print(f"\nOld Code Hunk ({len(example['old_hunk'])} chars):"
              f"\n{example['old_hunk'][:200]}{'...' if len(example['old_hunk']) > 200 else ''}")
        print(f"\nComment ({len(example['comment'])} chars):"
              f"\n{example['comment'][:200]}{'...' if len(example['comment']) > 200 else ''}")
        print(f"\nRefined Code ({len(example['new'])} chars):"
              f"\n{example['new'][:200]}{'...' if len(example['new']) > 200 else ''}")
        print("-" * 40)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR / 'raw'),
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default='data/processed/task4',
                       help='输出目录路径')
    parser.add_argument('--create_samples', action='store_true',
                       help='创建采样数据集')
    parser.add_argument('--train_sample_size', type=int, default=10000,
                       help='训练集采样大小')
    parser.add_argument('--valid_sample_size', type=int, default=2000,
                       help='验证集采样大小')
    parser.add_argument('--test_sample_size', type=int, default=1000,
                       help='测试集采样大小')
    parser.add_argument('--sample_strategy', choices=['random', 'balanced'], 
                       default='balanced', help='采样策略')
    parser.add_argument('--visualize', action='store_true',
                       help='创建可视化')
    
    args = parser.parse_args()
    
    # 分析数据
    data_files = {
        'train': os.path.join(args.data_dir, 'ref-train.jsonl'),
        'valid': os.path.join(args.data_dir, 'ref-valid.jsonl'),
        'test': os.path.join(args.data_dir, 'ref-test.jsonl')
    }
    
    all_examples = {}
    all_stats = {}
    
    for split, file_path in data_files.items():
        if os.path.exists(file_path):
            examples, stats = load_and_analyze_data(file_path)
            all_examples[split] = examples
            all_stats[split] = stats
            print_statistics(stats, split.upper())
            
            # 显示示例
            show_examples(examples, n=2)
        else:
            print(f"Warning: {file_path} not found")
    
    # 创建采样数据集
    if args.create_samples:
        os.makedirs(args.output_dir, exist_ok=True)
        
        sample_sizes = {
            'train': args.train_sample_size,
            'valid': args.valid_sample_size,
            'test': args.test_sample_size
        }
        
        for split, examples in all_examples.items():
            if examples:
                sample_file = os.path.join(args.output_dir, f'{split}_sample.jsonl')
                create_sample_dataset(
                    examples, 
                    sample_file, 
                    sample_sizes[split],
                    args.sample_strategy
                )
    
    # 创建可视化
    if args.visualize and len(all_stats) >= 3:
        try:
            visualize_statistics(
                all_stats['train'],
                all_stats['valid'], 
                all_stats['test'],
                os.path.join(args.output_dir, 'visualizations')
            )
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    # 保存统计信息
    stats_file = os.path.join(args.output_dir, 'data_statistics.json')
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\nStatistics saved to: {stats_file}")
    print(f"Preprocessing completed!")

if __name__ == '__main__':
    main()