"""
Level 3 评估指标模块

严格使用 utils/metrics.py 中的标准实现，确保与实验要求完全一致：
- 任务一：Accuracy, Precision, Recall, F1-Score (Macro)
- 任务二：Accuracy, Precision, Recall, F1-Score, MRR
- 任务三：BLEU-4, ROUGE-L, BERTScore
- 任务四：Exact Match, CodeBLEU
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到路径，以便导入 utils
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 从标准评估模块导入所有评估函数
from utils.metrics import (
    # 基础指标
    calculate_accuracy,
    calculate_precision_recall_f1,
    calculate_mrr,
    # 文本生成指标
    calculate_bleu,
    calculate_rouge_l,
    calculate_bert_score,
    # 代码评估指标
    calculate_exact_match,
    calculate_code_bleu,
    # 任务级别评估函数
    evaluate_task1,
    evaluate_task2,
    evaluate_task3,
    evaluate_task4,
)


def evaluate_quality_estimation(predictions: List[int],
                                references: List[int]) -> Dict[str, float]:
    """
    评估任务一：Diff Quality Estimation
    
    使用标准指标：Accuracy, Precision, Recall, F1-Score (Macro)
    
    Args:
        predictions: 预测标签列表 (0 或 1)
        references: 真实标签列表 (0 或 1)
    
    Returns:
        包含各指标的字典
    """
    return evaluate_task1(predictions, references)


def evaluate_code_localization(predictions: List[List[int]],
                               references: List[int]) -> Dict[str, float]:
    """
    评估任务二：Code Localization
    
    使用标准指标：Accuracy, Precision, Recall, F1-Score, MRR
    
    Args:
        predictions: 每个样本的预测排名列表 (每个元素是行号的排序列表)
        references: 真实的目标行号列表
    
    Returns:
        包含各指标的字典
    """
    return evaluate_task2(predictions, references)


def evaluate_comment_generation(
        predictions: List[str],
        references: List[str],
        use_bert_score: bool = True) -> Dict[str, float]:
    """
    评估任务三：Comment Generation
    
    使用标准指标：BLEU-4, ROUGE-L, BERTScore
    
    Args:
        predictions: 生成的评论列表
        references: 参考评论列表
        use_bert_score: 是否计算 BERTScore (计算较慢)
    
    Returns:
        包含各指标的字典
    """
    results = evaluate_task3(predictions, references)

    # 如果不需要 BERTScore，移除相关键
    if not use_bert_score and 'bert_score' in results:
        del results['bert_score']

    return results


def evaluate_code_refinement(predictions: List[str],
                             references: List[str],
                             lang: str = "python") -> Dict[str, float]:
    """
    评估任务四：Code Refinement
    
    使用标准指标：Exact Match, CodeBLEU
    
    Args:
        predictions: 修改后的代码列表
        references: 参考代码列表
        lang: 编程语言 (用于 CodeBLEU)
    
    Returns:
        包含各指标的字典
    """
    return evaluate_task4(predictions, references, lang=lang)


def evaluate_all_tasks(task: str, predictions: Any, references: Any,
                       **kwargs) -> Dict[str, float]:
    """
    统一评估入口
    
    Args:
        task: 任务名称 ('task1', 'task2', 'task3', 'task4')
        predictions: 预测结果
        references: 参考答案
        **kwargs: 额外参数
    
    Returns:
        包含各指标的字典
    """
    task_mapping = {
        'task1': evaluate_quality_estimation,
        'quality_estimation': evaluate_quality_estimation,
        'diff_quality_estimation': evaluate_quality_estimation,
        'task2': evaluate_code_localization,
        'code_localization': evaluate_code_localization,
        'task3': evaluate_comment_generation,
        'comment_generation': evaluate_comment_generation,
        'task4': evaluate_code_refinement,
        'code_refinement': evaluate_code_refinement,
    }

    task_key = task.lower().replace(' ', '_').replace('-', '_')

    if task_key not in task_mapping:
        raise ValueError(
            f"Unknown task: {task}. Available: {list(task_mapping.keys())}")

    return task_mapping[task_key](predictions, references, **kwargs)


# 为了向后兼容，保留一些别名
calculate_bleu_score = calculate_bleu
calculate_rouge_score = calculate_rouge_l
calculate_bertscore = calculate_bert_score
calculate_codebleu = calculate_code_bleu


def calculate_metrics(results: List[Dict], task_type: str) -> Dict[str, Any]:
    """
    根据任务类型计算评价指标
    
    Args:
        results: 结果列表，每个元素包含 result 和 input
        task_type: 任务类型 ('quality', 'comment', 'refinement')
    
    Returns:
        包含各评估指标的字典
    """
    import json

    if not results:
        return {"error": "No results to evaluate"}

    if task_type == "quality":
        # 任务一：Diff Quality Estimation
        # 提取预测和真实标签
        predictions = []
        references = []

        for r in results:
            result = r.get("result", {})
            input_data = r.get("input", {})

            # 获取预测值 - 从 task1_output 或直接结果中提取
            if isinstance(result, dict):
                if "task1_output" in result:
                    pred = result["task1_output"].get("needs_review", 1)
                elif "needs_review" in result:
                    pred = result.get("needs_review", 1)
                else:
                    pred = 1  # 默认需要评审
            else:
                pred = 1

            # 获取真实标签
            label = input_data.get("quality_label",
                                   input_data.get("label", -1))

            if label != -1:  # 只有有标签的才计入
                predictions.append(int(pred))
                references.append(int(label))

        if not predictions:
            return {"error": "No valid samples with labels"}

        # 使用标准评估函数
        return evaluate_quality_estimation(predictions, references)

    elif task_type == "comment":
        # 任务三：Comment Generation
        predictions = []
        references = []

        for r in results:
            result = r.get("result", {})
            input_data = r.get("input", {})

            # 获取生成的评审意见
            if isinstance(result, dict):
                pred = result.get("review_comment", "")
            else:
                pred = str(result) if result else ""

            # 获取参考答案
            ref = input_data.get("comment",
                                 input_data.get("review_comment", ""))

            if pred and ref:
                predictions.append(pred)
                references.append(ref)

        if not predictions:
            return {
                "error": "No valid samples for comment generation evaluation"
            }

        return evaluate_comment_generation(predictions,
                                           references,
                                           use_bert_score=True)

    elif task_type == "refinement":
        # 任务四：Code Refinement
        predictions = []
        references = []

        for r in results:
            result = r.get("result", {})
            input_data = r.get("input", {})

            # 获取生成的代码
            if isinstance(result, dict):
                pred = result.get("fixed_code", result.get("new_code", ""))
            else:
                pred = str(result) if result else ""

            # 获取参考代码
            ref = input_data.get("new_code", input_data.get("fixed_code", ""))

            if pred and ref:
                predictions.append(pred)
                references.append(ref)

        if not predictions:
            return {"error": "No valid samples for code refinement evaluation"}

        # 检测语言
        lang = input_data.get("language", "python") if input_data else "python"

        return evaluate_code_refinement(predictions, references, lang=lang)

    else:
        return {"error": f"Unknown task type: {task_type}"}


def print_metrics(metrics: Dict[str, Any], task_type: str) -> None:
    """
    打印评价指标
    
    Args:
        metrics: 评估结果字典
        task_type: 任务类型
    """
    task_names = {
        "quality": "任务一：Diff Quality Estimation",
        "comment": "任务三：Comment Generation",
        "refinement": "任务四：Code Refinement",
    }

    print(f"\n{'='*60}")
    print(f"{task_names.get(task_type, task_type)} 评估结果")
    print(f"{'='*60}")

    if "error" in metrics:
        print(f"[x] 错误: {metrics['error']}")
        return

    # 根据任务类型打印对应的指标
    if task_type == "quality":
        # 任务一指标：Accuracy, Precision, Recall, F1-Score (Macro)
        print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0):.4f}")
        print(f"F1-Score:  {metrics.get('f1', 0):.4f}")

    elif task_type == "comment":
        # 任务三指标：BLEU-4, ROUGE-L, BERTScore
        print(f"BLEU-4:    {metrics.get('bleu', 0):.4f}")
        print(f"ROUGE-L:   {metrics.get('rouge_l', 0):.4f}")
        if 'bert_score' in metrics:
            print(f"BERTScore: {metrics.get('bert_score', 0):.4f}")

    elif task_type == "refinement":
        # 任务四指标：Exact Match, CodeBLEU
        print(f"Exact Match: {metrics.get('exact_match', 0):.4f}")
        print(f"CodeBLEU:    {metrics.get('code_bleu', 0):.4f}")

    else:
        # 通用打印
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    print(f"{'='*60}\n")


def save_metrics(metrics: Dict[str, Any], task_type: str,
                 output_dir: Path) -> None:
    """
    保存评价指标到文件
    
    Args:
        metrics: 评估结果字典
        task_type: 任务类型或文件名前缀
        output_dir: 输出目录
    """
    import json
    from datetime import datetime

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 添加时间戳
    metrics_with_meta = {
        "timestamp": datetime.now().isoformat(),
        "task": task_type,
        **metrics
    }

    output_file = output_dir / f"{task_type}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_with_meta, f, indent=2, ensure_ascii=False)
