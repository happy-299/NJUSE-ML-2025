"""
评估指标计算工具

包含各任务所需的评估指标实现:
- 任务一：Accuracy, Precision, Recall, F1-Score (Macro)
- 任务二：Accuracy, Precision, Recall, F1-Score, MRR
- 任务三：BLEU-4, ROUGE-L, BERTScore
- 任务四：Exact Match, CodeBLEU
"""

import numpy as np
from typing import List, Dict, Any, Union
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# ==================== 分类任务指标 (任务一、任务二) ====================

def calculate_accuracy(predictions: List[int], references: List[int]) -> float:
    """计算准确率"""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions)


def calculate_precision_recall_f1(predictions: List[int], references: List[int], 
                                 average: str = 'macro') -> Dict[str, float]:
    """计算精确率、召回率和F1分数
    
    Args:
        predictions: 预测结果列表
        references: 真实标签列表
        average: 'macro', 'micro', 'weighted' 或 'binary'
    
    Returns:
        包含precision, recall, f1的字典
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    # 获取所有类别
    all_labels = sorted(set(predictions + references))
    
    if average == 'binary' and len(all_labels) != 2:
        raise ValueError("Binary average requires exactly 2 classes")
    
    # 计算每个类别的TP, FP, FN
    class_metrics = {}
    for label in all_labels:
        tp = sum(1 for p, r in zip(predictions, references) if p == label and r == label)
        fp = sum(1 for p, r in zip(predictions, references) if p == label and r != label)
        fn = sum(1 for p, r in zip(predictions, references) if p != label and r == label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for r in references if r == label)
        }
    
    # 根据average参数计算最终结果
    if average == 'macro':
        precision = np.mean([metrics['precision'] for metrics in class_metrics.values()])
        recall = np.mean([metrics['recall'] for metrics in class_metrics.values()])
        f1 = np.mean([metrics['f1'] for metrics in class_metrics.values()])
    elif average == 'micro':
        total_tp = sum(1 for p, r in zip(predictions, references) if p == r)
        total_fp = sum(1 for p, r in zip(predictions, references) if p != r)
        total_fn = total_fp  # 在多分类中，FP = FN
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    elif average == 'weighted':
        total_support = sum(metrics['support'] for metrics in class_metrics.values())
        precision = sum(metrics['precision'] * metrics['support'] 
                       for metrics in class_metrics.values()) / total_support
        recall = sum(metrics['recall'] * metrics['support'] 
                    for metrics in class_metrics.values()) / total_support
        f1 = sum(metrics['f1'] * metrics['support'] 
                for metrics in class_metrics.values()) / total_support
    elif average == 'binary':
        # 对于二分类，通常取正类(标签1)的指标
        positive_label = max(all_labels)
        precision = class_metrics[positive_label]['precision']
        recall = class_metrics[positive_label]['recall']
        f1 = class_metrics[positive_label]['f1']
    else:
        raise ValueError(f"Unknown average: {average}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics
    }


def calculate_mrr(predictions: List[List[int]], references: List[int]) -> float:
    """计算平均倒数排名 (MRR)
    
    用于任务二：问题代码定位
    
    Args:
        predictions: 每个样本的预测排序列表 (按置信度降序)
        references: 真实的正确答案位置
        
    Returns:
        MRR分数
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    reciprocal_ranks = []
    for pred_list, true_label in zip(predictions, references):
        if true_label in pred_list:
            rank = pred_list.index(true_label) + 1  # 排名从1开始
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)


# ==================== 文本生成任务指标 (任务三、任务四) ====================

def calculate_bleu(predictions: List[str], references: List[str], n_gram: int = 4) -> float:
    """计算BLEU分数
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
        n_gram: N-gram的最大长度
        
    Returns:
        BLEU分数
    """
    try:
        import nltk
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        
        # 确保nltk数据已下载
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # 分词
        refs = [[ref.split()] for ref in references]
        preds = [pred.split() for pred in predictions]
        
        # 使用平滑函数避免零分
        smoothie = SmoothingFunction().method4
        
        # 计算BLEU
        if n_gram == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        elif n_gram == 2:
            weights = (0.5, 0.5)
        elif n_gram == 1:
            weights = (1.0,)
        else:
            weights = tuple([1.0/n_gram] * n_gram)
        
        bleu_score = corpus_bleu(refs, preds, weights=weights, smoothing_function=smoothie)
        return bleu_score
        
    except ImportError:
        logger.warning("NLTK not available, using simplified BLEU calculation")
        return _simple_bleu(predictions, references, n_gram)


def _simple_bleu(predictions: List[str], references: List[str], n_gram: int = 4) -> float:
    """简化版BLEU计算 (不依赖NLTK)"""
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    total_score = 0.0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        scores = []
        for n in range(1, n_gram + 1):
            pred_ngrams = Counter(get_ngrams(pred_tokens, n))
            ref_ngrams = Counter(get_ngrams(ref_tokens, n))
            
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total_pred = sum(pred_ngrams.values())
            
            if total_pred > 0:
                scores.append(overlap / total_pred)
            else:
                scores.append(0.0)
        
        # 几何平均
        if all(s > 0 for s in scores):
            total_score += np.exp(np.mean(np.log(scores)))
        else:
            total_score += 0.0
    
    return total_score / len(predictions)


def calculate_rouge_l(predictions: List[str], references: List[str]) -> float:
    """计算ROUGE-L分数
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
        
    Returns:
        ROUGE-L F1分数
    """
    try:
        from rouge import Rouge
        rouge = Rouge()
        
        # 过滤空字符串
        valid_pairs = [(p, r) for p, r in zip(predictions, references) 
                      if p.strip() and r.strip()]
        
        if not valid_pairs:
            return 0.0
        
        preds, refs = zip(*valid_pairs)
        scores = rouge.get_scores(list(preds), list(refs), avg=True)
        return scores['rouge-l']['f']
        
    except ImportError:
        logger.warning("rouge package not available, using simplified ROUGE-L calculation")
        return _simple_rouge_l(predictions, references)


def _simple_rouge_l(predictions: List[str], references: List[str]) -> float:
    """简化版ROUGE-L计算"""
    def lcs_length(seq1, seq2):
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    total_f1 = 0.0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        if not pred_tokens or not ref_tokens:
            total_f1 += 0.0
            continue
        
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        
        precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        total_f1 += f1
    
    return total_f1 / len(predictions)


def calculate_bert_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """计算BERTScore
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
        
    Returns:
        包含precision, recall, f1的字典
    """
    try:
        from bert_score import score
        
        P, R, F1 = score(predictions, references, lang="en", verbose=False)
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
        
    except ImportError:
        logger.warning("bert_score package not available")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


def calculate_exact_match(predictions: List[str], references: List[str]) -> float:
    """计算完全匹配率
    
    用于任务四：修复代码生成
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
        
    Returns:
        完全匹配率
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    matches = [pred.strip() == ref.strip() for pred, ref in zip(predictions, references)]
    return np.mean(matches)


def calculate_code_bleu(predictions: List[str], references: List[str], 
                       language: str = 'java') -> Dict[str, float]:
    """计算CodeBLEU分数
    
    CodeBLEU结合了语法匹配、数据流匹配和传统BLEU
    
    Args:
        predictions: 预测代码列表
        references: 参考代码列表  
        language: 编程语言
        
    Returns:
        包含codebleu分数的字典
    """
    try:
        from codebleu import calc_codebleu
        
        result = calc_codebleu(references, predictions, lang=language, weights=(0.25, 0.25, 0.25, 0.25))
        return result
        
    except ImportError:
        logger.warning("codebleu package not available, using BLEU-4 as approximation")
        bleu4 = calculate_bleu(predictions, references, n_gram=4)
        return {'codebleu': bleu4}
    except Exception as e:
        logger.warning(f"CodeBLEU calculation failed: {e}, using BLEU-4 as approximation")
        bleu4 = calculate_bleu(predictions, references, n_gram=4)
        return {'codebleu': bleu4}


# ==================== 综合评估函数 ====================

def evaluate_task1(predictions: List[int], references: List[int]) -> Dict[str, float]:
    """任务一评估：代码质量评估 (二分类)"""
    results = {}
    
    # 基本指标
    results['accuracy'] = calculate_accuracy(predictions, references)
    
    # 精确率、召回率、F1
    prf_results = calculate_precision_recall_f1(predictions, references, average='macro')
    results.update(prf_results)
    
    return results


def evaluate_task2(predictions: List[List[int]], references: List[int],
                   flat_predictions: List[int] = None) -> Dict[str, float]:
    """任务二评估：问题代码定位"""
    results = {}
    
    # MRR (主要指标)
    results['mrr'] = calculate_mrr(predictions, references)
    
    # 如果提供了扁平化的预测结果，计算分类指标
    if flat_predictions is not None:
        results['accuracy'] = calculate_accuracy(flat_predictions, references)
        prf_results = calculate_precision_recall_f1(flat_predictions, references, average='macro')
        results.update(prf_results)
    
    return results


def evaluate_task3(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """任务三评估：评审意见生成"""
    results = {}
    
    # BLEU-4
    results['bleu4'] = calculate_bleu(predictions, references, n_gram=4)
    
    # ROUGE-L
    results['rouge_l'] = calculate_rouge_l(predictions, references)
    
    # BERTScore
    bert_results = calculate_bert_score(predictions, references)
    results['bert_score_f1'] = bert_results['f1']
    results['bert_score_precision'] = bert_results['precision']
    results['bert_score_recall'] = bert_results['recall']
    
    return results


def evaluate_task4(predictions: List[str], references: List[str], 
                   languages: List[str] = None) -> Dict[str, float]:
    """任务四评估：修复代码生成"""
    results = {}
    
    # Exact Match
    results['exact_match'] = calculate_exact_match(predictions, references)
    
    # CodeBLEU
    if languages:
        # 按语言分组计算CodeBLEU
        from collections import defaultdict
        lang_groups = defaultdict(list)
        for pred, ref, lang in zip(predictions, references, languages):
            lang_groups[lang].append((pred, ref))
        
        codebleu_scores = []
        for lang, items in lang_groups.items():
            if len(items) >= 5:  # 只计算样本数足够的语言
                lang_preds, lang_refs = zip(*items)
                try:
                    codebleu_result = calculate_code_bleu(list(lang_preds), list(lang_refs), lang)
                    codebleu_scores.append(codebleu_result['codebleu'])
                except:
                    continue
        
        if codebleu_scores:
            results['codebleu'] = np.mean(codebleu_scores)
        else:
            results['codebleu'] = 0.0
    else:
        # 使用默认语言
        codebleu_result = calculate_code_bleu(predictions, references, 'java')
        results['codebleu'] = codebleu_result['codebleu']
    
    # BLEU-4 作为补充
    results['bleu4'] = calculate_bleu(predictions, references, n_gram=4)
    
    return results


# ==================== 工具函数 ====================

def print_evaluation_results(results: Dict[str, float], task_name: str):
    """打印评估结果"""
    print(f"\n{'='*50}")
    print(f"{task_name} Evaluation Results")
    print(f"{'='*50}")
    
    for metric, score in results.items():
        if isinstance(score, dict):
            continue  # 跳过嵌套字典
        print(f"{metric.upper().replace('_', ' ')}: {score:.4f}")


def save_evaluation_results(results: Dict[str, Any], output_file: str):
    """保存评估结果到JSON文件"""
    import json
    import os
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # 测试评估函数
    print("Testing evaluation metrics...")
    
    # 测试任务一 (二分类)
    pred1 = [0, 1, 1, 0, 1]
    ref1 = [0, 1, 0, 0, 1]
    results1 = evaluate_task1(pred1, ref1)
    print_evaluation_results(results1, "Task 1 (Code Quality Estimation)")
    
    # 测试任务三 (文本生成)
    pred3 = ["This is a prediction", "Another prediction"]
    ref3 = ["This is a reference", "Another reference"]
    results3 = evaluate_task3(pred3, ref3)
    print_evaluation_results(results3, "Task 3 (Comment Generation)")
    
    # 测试任务四 (代码生成)
    pred4 = ["def func():\n    return True", "int x = 5;"]
    ref4 = ["def func():\n    return False", "int x = 10;"]
    results4 = evaluate_task4(pred4, ref4, ["python", "java"])
    print_evaluation_results(results4, "Task 4 (Code Refinement)")
    
    print("\nAll tests completed!")