"""
Level 3 - AI Agent 代码审查主程序
命令行接口
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Optional

from .config import get_config
from .data_loader import DataLoader
from .metrics import calculate_metrics, print_metrics, save_metrics

# 延迟导入workflow以避免不必要的依赖加载
# from .workflows.review_workflow import CodeReviewWorkflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def review_code(code: str,
                language: str = "python",
                diff: Optional[str] = None):
    """
    审查单个代码片段
    
    Args:
        code: 代码内容
        language: 编程语言
        diff: 代码差异
    """
    from .workflows.review_workflow import CodeReviewWorkflow

    logger.info("初始化代码审查工作流...")
    workflow = CodeReviewWorkflow()

    logger.info("开始审查...")
    result = workflow.review(code=code, language=language, diff=diff)

    # 打印结果
    print("\n" + "=" * 60)
    print("代码审查结果")
    print("=" * 60)

    if result.get("error"):
        print(f"\n[x] 错误: {result['error']}")
        return

    print(f"\n[质量评分] {result['quality_score']:.1f}/100")
    print(f"⚡ 优先级: {result['priority']}")

    issues = result.get("issues", [])
    if issues:
        print(f"\n🔍 发现 {len(issues)} 个问题:")
        for i, issue in enumerate(issues, 1):
            desc = issue.get('description', str(issue))
            print(f"  {i}. {desc}")
    else:
        print("\n[OK] 未发现明显问题")

    suggestions = result.get("suggestions", [])
    if suggestions:
        print(f"\n💡 {len(suggestions)} 条改进建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")

    print("\n" + "-" * 60)
    print("完整审查评论:")
    print("-" * 60)
    print(result.get("review_comment", ""))
    print("=" * 60 + "\n")


def review_file(file_path: str, language: Optional[str] = None):
    """
    审查文件
    
    Args:
        file_path: 文件路径
        language: 编程语言（自动检测）
    """
    path = Path(file_path)

    if not path.exists():
        print(f"[x] 文件不存在: {file_path}")
        return

    # 自动检测语言
    if language is None:
        ext = path.suffix.lower()
        lang_map = {
            '.py': 'python',
            '.java': 'java',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.cpp': 'cpp',
            '.c': 'c'
        }
        language = lang_map.get(ext, 'unknown')

    # 读取文件
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()

    print(f"\n📄 审查文件: {file_path}")
    print(f"🔤 语言: {language}")

    review_code(code, language)


def review_dataset(task_type: str = "quality",
                   limit: int = 100,
                   split: str = "test"):
    """
    审查数据集中的样本
    
    Args:
        task_type: 任务类型 (refinement/comment/quality)
        limit: 限制样本数量
        split: 数据集划分 (train/valid/test)
    """
    # 针对 quality 任务使用专门的 Task1Workflow
    if task_type == "quality":
        from .workflows.task1_workflow import Task1Workflow
        workflow_class = Task1Workflow
    else:
        from .workflows.review_workflow import CodeReviewWorkflow
        workflow_class = CodeReviewWorkflow

    logger.info(f"加载 {task_type} 数据集 ({split} split)...")

    loader = DataLoader()
    workflow = workflow_class()

    # 加载数据
    if task_type == "refinement":
        data = loader.load_code_refinement(split=split, limit=limit)
    elif task_type == "comment":
        data = loader.load_comment_generation(split=split, limit=limit)
    elif task_type == "quality":
        data = loader.load_diff_quality(split=split, limit=limit)
    else:
        print(f"[x] 未知任务类型: {task_type}")
        return

    print(f"\n[加载] 加载了 {len(data)} 个样本 (来自 {split} set)")

    results = []

    for i, item in enumerate(data, 1):
        print(f"\n{'=' * 60}")
        print(f"样本 {i}/{len(data)}")
        print(f"{'=' * 60}")

        # 准备输入
        review_input = loader.prepare_review_input(item, task_type)

        # 执行审查
        try:
            if task_type == "quality":
                # 使用 Task1Workflow 的简化接口
                result_data = workflow.run(code=review_input.get("code", ""),
                                           diff=review_input.get("diff", ""),
                                           language=review_input.get(
                                               "language", "unknown"))
                result = {
                    "result": result_data.get("task1_output", {}),
                    "error": result_data.get("error")
                }
            else:
                # 使用原来的 CodeReviewWorkflow
                result = workflow.review(
                    code=review_input.get("code")
                    or review_input.get("old_code", ""),
                    language=review_input.get("language", "unknown"),
                    diff=review_input.get("diff"),
                    task_type=task_type,
                    context=review_input.get("context", {}))

            results.append({
                "sample_id": i,
                "result": result.get("result", result),  # 兼容两种格式
                "input": review_input
            })

            # 打印简要结果
            if task_type == "quality":
                task1_output = result.get("result", {})
                needs_review = task1_output.get("needs_review", 1)
                confidence = task1_output.get("confidence", 0.0)
                print(f"\n[OK] 评审必要性: {'需要' if needs_review else '不需要'}")
                print(f"[OK] 置信度: {confidence:.2f}")
                print(f"[OK] 发现问题: {len(task1_output.get('issues', []))} 个")
            else:
                print(f"\n[OK] 质量评分: {result.get('quality_score', 'N/A')}")
                print(f"[OK] 发现问题: {len(result.get('issues', []))} 个")
                print(f"[OK] 改进建议: {len(result.get('suggestions', []))} 条")

        except Exception as e:
            logger.error(f"审查失败: {e}")
            print(f"\n[x] 错误: {e}")

    # 保存结果到项目根目录的 outputs/level3，按任务类型组织
    project_root = Path(__file__).parent.parent.parent  # NJUSE-ML-2025/

    # 确定任务编号
    task_num = "task3"  # 默认任务三（评审生成）
    if task_type == "quality":
        task_num = "task1"
    elif task_type == "refinement":
        task_num = "task4"

    output_dir = project_root / "outputs" / "level3" / task_num
    output_dir.mkdir(parents=True, exist_ok=True)

    # Level 2 标准输出格式：predictions.json
    # 格式: [{idx, sample_id, prediction, confidence, reasoning, ground_truth, raw_response}, ...]
    predictions = []
    for idx, r in enumerate(results):
        task1_output = r["result"].get("task1_output", {})
        predictions.append({
            "idx":
            idx,
            "sample_id":
            r["sample_id"],
            "prediction":
            task1_output.get("needs_review", 1),  # 0 或 1
            "confidence":
            task1_output.get("confidence", 0.0),
            "reasoning":
            task1_output.get("reasoning", ""),
            "ground_truth":
            r["input"].get("quality_label", -1),
            "raw_response":
            task1_output.get("raw_response", ""),
        })

    output_file = output_dir / f"predictions_{split}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\n[保存] 结果已保存到: {output_file}")

    # 计算并输出评价指标
    logger.info("计算评价指标...")
    metrics = calculate_metrics(results, task_type)
    print_metrics(metrics, task_type)

    # 持久化评价指标到文件
    save_metrics(metrics, task_type, output_dir)
    print(f"[指标] 指标已保存到: {output_dir / f'metrics_{task_type}.json'}")


def show_stats():
    """显示数据集统计信息"""
    loader = DataLoader()
    stats = loader.get_dataset_stats()

    print("\n" + "=" * 60)
    print("数据集统计信息")
    print("=" * 60)

    for task, splits in stats.items():
        print(f"\n[Metrics] {task.replace('_', ' ').title()}:")
        for split, count in splits.items():
            print(f"  • {split:10s}: {count:6d} 条")

    print("=" * 60 + "\n")


def review_dataset_full_pipeline(task_type: str = "quality",
                                 limit: int = 100,
                                 split: str = "test"):
    """
    使用完整四任务端到端流程审查数据集
    
    Args:
        task_type: 任务类型 (用于选择数据集，但会执行完整流程)
        limit: 限制样本数量
        split: 数据集划分 (train/valid/test)
    """
    from .workflows.full_pipeline_workflow import FullPipelineWorkflow

    logger.info(f"加载 {task_type} 数据集 ({split} split)，执行完整四任务流程...")

    loader = DataLoader()
    workflow = FullPipelineWorkflow()

    # 加载数据（优先使用quality数据集，因为有标签）
    if task_type == "quality":
        data = loader.load_diff_quality(split=split, limit=limit)
    elif task_type == "comment":
        data = loader.load_comment_generation(split=split, limit=limit)
    elif task_type == "refinement":
        data = loader.load_code_refinement(split=split, limit=limit)
    else:
        print(f"[x] 未知任务类型: {task_type}")
        return

    print(f"\n[加载] 加载了 {len(data)} 个样本 (来自 {split} set)")
    print("[流程] 将执行完整四任务流程：质量评估->问题定位->评审生成->代码修复\n")

    results = []

    for i, item in enumerate(data, 1):
        print(f"\n{'=' * 60}")
        print(f"样本 {i}/{len(data)}")
        print(f"{'=' * 60}")

        try:
            # 准备输入
            review_input = loader.prepare_review_input(item, task_type)

            # 执行完整四任务流程
            result = workflow.review(code=review_input.get("code", ""),
                                     language=review_input.get(
                                         "language", "unknown"),
                                     diff=review_input.get("diff", ""),
                                     old_code=review_input.get("old_code", ""),
                                     new_code=review_input.get("new_code", ""),
                                     context=review_input.get("context", {}))

            results.append({
                "sample_id": i,
                "result": result,
                "input": review_input
            })

            # 打印四任务结果
            print(
                f"\n[OK] 任务一 - 质量评估: 评分={result.get('quality_score', 'N/A')}, 需要评审={'是' if result.get('needs_review') else '否'}"
            )
            print(
                f"[OK] 任务二 - 问题定位: 发现 {len(result.get('problem_locations') or [])} 个问题位置"
            )
            print(
                f"[OK] 任务三 - 评审意见: {(result.get('review_comment') or 'N/A')[:80]}..."
            )
            print(
                f"[OK] 任务四 - 代码修复: 生成 {len(result.get('fixed_code') or '')} 字符"
            )

        except Exception as e:
            import traceback
            logger.error(f"审查失败: {e}")
            logger.error(f"完整错误:\n{traceback.format_exc()}")
            print(f"\n[x] 错误: {e}")
            print(f"详细信息: {traceback.format_exc()}")

    # 保存结果到项目根目录的 outputs/level3
    project_root = Path(__file__).parent.parent.parent  # NJUSE-ML-2025/
    output_dir = project_root / "outputs" / "level3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整的模型回答和统计数据
    detailed_results = []
    simplified_results = []

    for r in results:
        # 提取Agent决策信息（安全处理None值）
        task1_output = r["result"].get("task1_output") or {}
        task4_output = r["result"].get("task4_output") or {}
        complexity = task1_output.get("code_features",
                                      {}).get("complexity", "N/A")

        # 安全获取字段（防止None导致len()报错）
        review_comment = r["result"].get("review_comment") or ""
        fixed_code = r["result"].get("fixed_code") or ""
        problem_locations = r["result"].get("problem_locations") or []

        # 完整的模型回答（用于详细分析）
        detailed_results.append({
            "sample_id": r["sample_id"],
            "input": r["input"],  # 原始输入
            "task1_quality_assessment": {
                "quality_score": r["result"].get("quality_score"),
                "needs_review": r["result"].get("needs_review"),
                "code_complexity": complexity,
                "model_output": r["result"].get("quality_score")  # 模型原始输出
            },
            "task2_problem_localization": {
                "problem_locations": problem_locations,
                "problem_count": len(problem_locations),
                "model_output": problem_locations  # 模型原始输出
            },
            "task3_review_generation": {
                "review_comment": review_comment,
                "review_length": len(review_comment),
                "model_output": review_comment  # 模型原始输出
            },
            "task4_code_fixing": {
                "fixed_code": fixed_code,
                "fixed_code_length": len(fixed_code),
                "verified": task4_output.get("verified", False),
                "retry_count": task4_output.get("retry_count", 0),
                "model_output": fixed_code  # 模型原始输出
            },
            "agent_decisions": {
                "task1_to_task2":
                "continue" if r["result"].get("needs_review") else "skip",
                "task2_to_task3":
                "continue" if len(problem_locations) > 0 else "skip",
                "task4_verification":
                "passed" if task4_output.get("verified", False) else "failed",
                "task4_retries":
                task4_output.get("retry_count", 0)
            }
        })

        # 简化统计数据（用于快速查看）
        simplified_results.append({
            "sample_id":
            r["sample_id"],
            "task1_quality_score":
            r["result"].get("quality_score"),
            "task1_needs_review":
            r["result"].get("needs_review"),
            "task1_code_complexity":
            complexity,
            "task2_problem_count":
            len(problem_locations),
            "task3_review_length":
            len(review_comment),
            "task4_fixed_code_length":
            len(fixed_code),
            "task4_verified":
            task4_output.get("verified", False),
            "task4_retry_count":
            task4_output.get("retry_count", 0)
        })

    # 保存详细结果（包含完整模型回答） - 保存到 all 目录
    output_dir_all = output_dir / "all"
    output_dir_all.mkdir(parents=True, exist_ok=True)

    detailed_file = output_dir_all / f"detailed_results_{split}.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"\n[保存] 详细结果已保存到: {detailed_file}")

    # 保存简化结果（统计数据）
    summary_file = output_dir_all / f"summary_results_{split}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(simplified_results, f, indent=2, ensure_ascii=False)
    print(f"[保存] 统计摘要已保存到: {summary_file}")

    # 计算四个任务的评价指标
    logger.info("计算四任务评价指标...")

    # 任务一：质量评估 - 保存到 task1 目录
    from .metrics import calculate_metrics as calc_metrics, save_metrics, print_metrics

    output_dir_task1 = output_dir / "task1"
    output_dir_task1.mkdir(parents=True, exist_ok=True)

    metrics_task1 = calc_metrics(results, "quality")
    print_metrics(metrics_task1, "quality")
    save_metrics(metrics_task1, f"metrics_{split}", output_dir_task1)

    # 任务二：问题定位 - 保存到 task2 目录
    output_dir_task2 = output_dir / "task2"
    output_dir_task2.mkdir(parents=True, exist_ok=True)

    metrics_task2 = calc_metrics(results, "refinement")
    print_metrics(metrics_task2, "refinement")
    save_metrics(metrics_task2, f"metrics_{split}", output_dir_task2)

    # 任务三：评审意见生成
    # 尝试使用标准评估指标 (BLEU-4, ROUGE-L, BERTScore)
    output_dir_task3 = output_dir / "task3"
    output_dir_task3.mkdir(parents=True, exist_ok=True)

    # 检查是否有 ground truth 评审意见
    metrics_task3 = calc_metrics(results, "comment")

    if "error" not in metrics_task3:
        # 有 ground truth，使用标准指标
        print_metrics(metrics_task3, "comment")
        save_metrics(metrics_task3, f"metrics_{split}", output_dir_task3)
        review_stats = metrics_task3
    else:
        # 没有 ground truth，只统计生成情况
        review_stats = {
            "total_samples":
            len(results),
            "generated_reviews":
            sum(1 for r in results
                if (r.get("result") or {}).get("review_comment")),
            "avg_review_length":
            sum(
                len((r.get("result") or {}).get("review_comment") or "")
                for r in results) / len(results) if results else 0,
            "generation_rate":
            sum(1 for r in results
                if (r.get("result") or {}).get("review_comment")) /
            len(results) * 100 if results else 0,
            "note":
            "无 ground truth，仅统计生成情况",
            "task":
            "任务三：评审意见生成（统计）"
        }
        print(f"\n{'='*60}")
        print(f"任务三：评审意见生成（统计 - 无ground truth）")
        print(f"{'='*60}")
        print(f"总样本数: {review_stats['total_samples']}")
        print(f"成功生成: {review_stats['generated_reviews']}")
        print(f"生成率: {review_stats['generation_rate']:.1f}%")
        print(f"平均长度: {review_stats['avg_review_length']:.1f} 字符")
        print(f"{'='*60}\n")
        save_metrics(review_stats, f"metrics_{split}", output_dir_task3)

    # 任务四：代码修复
    # 尝试使用标准评估指标 (Exact Match, CodeBLEU)
    output_dir_task4 = output_dir / "task4"
    output_dir_task4.mkdir(parents=True, exist_ok=True)

    # 检查是否有 ground truth 修复代码
    metrics_task4 = calc_metrics(results, "refinement")

    if "error" not in metrics_task4:
        # 有 ground truth，使用标准指标
        print_metrics(metrics_task4, "refinement")
        save_metrics(metrics_task4, f"metrics_{split}", output_dir_task4)
        fixing_stats = metrics_task4
    else:
        # 没有 ground truth，只统计生成情况
        fixing_stats = {
            "total_samples":
            len(results),
            "generated_fixes":
            sum(1 for r in results
                if (r.get("result") or {}).get("fixed_code")),
            "verified_fixes":
            sum(1 for r in results
                if ((r.get("result") or {}).get("task4_output") or {}
                    ).get("verified", False)),
            "avg_fixed_length":
            sum(
                len((r.get("result") or {}).get("fixed_code") or "")
                for r in results) / len(results) if results else 0,
            "generation_rate":
            sum(1 for r in results
                if (r.get("result") or {}).get("fixed_code")) / len(results) *
            100 if results else 0,
            "verification_rate":
            sum(1 for r in results if (
                (r.get("result") or {}).get("task4_output") or {}
            ).get("verified", False)) / len(results) * 100 if results else 0,
            "avg_retry_count":
            sum(((r.get("result") or {}).get("task4_output") or {}
                 ).get("retry_count", 0)
                for r in results) / len(results) if results else 0,
            "note":
            "无 ground truth，仅统计生成情况",
            "task":
            "任务四：代码修复（统计）"
        }
        print(f"\n{'='*60}")
        print(f"任务四：代码修复（统计 - 无ground truth）")
        print(f"{'='*60}")
        print(f"总样本数: {fixing_stats['total_samples']}")
        print(f"成功生成: {fixing_stats['generated_fixes']}")
        print(f"通过验证: {fixing_stats['verified_fixes']}")
        print(f"生成率: {fixing_stats['generation_rate']:.1f}%")
        print(f"验证率: {fixing_stats['verification_rate']:.1f}%")
        print(f"平均长度: {fixing_stats['avg_fixed_length']:.1f} 字符")
        print(f"平均重试: {fixing_stats['avg_retry_count']:.2f} 次")
        print(f"{'='*60}\n")
        save_metrics(fixing_stats, f"metrics_{split}", output_dir_task4)

    # 保存综合指标到 all 目录
    from datetime import datetime
    comprehensive_metrics = {
        "task1_quality_assessment": metrics_task1,
        "task2_problem_localization": metrics_task2,
        "task3_review_generation": review_stats,
        "task4_code_fixing": fixing_stats,
        "overall_statistics": {
            "total_samples": len(results),
            "dataset": task_type,
            "split": split,
            "timestamp": datetime.now().isoformat()
        }
    }
    comprehensive_file = output_dir_all / f"comprehensive_metrics_{split}.json"
    with open(comprehensive_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[指标] 综合指标已保存到: {comprehensive_file}")
    print(f"[指标] 各任务指标已保存到 outputs/level3/ 目录的各子目录")
    print(f"  - task1/: 任务一（代码质量评估）")
    print(f"  - task2/: 任务二（问题代码定位）")
    print(f"  - task3/: 任务三（评审意见生成）")
    print(f"  - task4/: 任务四（代码修复）")
    print(f"  - all/: 完整流程综合结果")


def show_stats():
    """显示数据集统计信息"""
    loader = DataLoader()
    stats = loader.get_dataset_stats()

    print("\n" + "=" * 60)
    print("数据集统计信息")
    print("=" * 60)

    for task, splits in stats.items():
        print(f"\n[Metrics] {task.replace('_', ' ').title()}:")
        for split, count in splits.items():
            print(f"  • {split:10s}: {count:6d} 条")

    print("=" * 60 + "\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Level 3 - AI Agent 代码审查系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 审查代码片段
  python -m lab3.level3_agent.main --code "def add(a,b): return a+b"
  
  # 审查文件
  python -m lab3.level3_agent.main --file script.py
  
  # 审查数据集样本
  python -m lab3.level3_agent.main --dataset refinement --limit 5
  
  # 显示数据集统计
  python -m lab3.level3_agent.main --stats
""")

    parser.add_argument('--code', type=str, help='要审查的代码片段')

    parser.add_argument('--file', type=str, help='要审查的代码文件路径')

    parser.add_argument('--language',
                        type=str,
                        default='python',
                        help='编程语言 (默认: python)')

    parser.add_argument('--diff', type=str, help='代码差异（可选）')

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['task1', 'task2', 'task3', 'task4', 'all'],
        default='task1',
        help=
        '任务类型: task1(质量评估), task2(问题定位), task3(评审生成), task4(代码修复), all(完整流程)')

    parser.add_argument('--limit',
                        type=int,
                        default=100,
                        help='数据集样本数量限制 (默认: 100)')

    parser.add_argument('--split',
                        type=str,
                        choices=['train', 'valid', 'test'],
                        default='test',
                        help='数据集划分 (默认: test)')

    parser.add_argument('--full-pipeline',
                        action='store_true',
                        help='使用完整四任务端到端流程（等同于 --dataset all）')

    parser.add_argument(
        '--source-dataset',
        type=str,
        choices=['quality', 'comment', 'refinement'],
        default='refinement',
        help='完整流程的数据源: quality(有任务一标签), comment(有任务三标签), refinement(有任务四标签，默认)'
    )

    parser.add_argument('--stats', action='store_true', help='显示数据集统计信息')

    parser.add_argument('--config', type=str, help='配置文件路径 (默认: lab3/.env)')

    parser.add_argument('--llm-type',
                        type=str,
                        choices=['local', 'openai', 'ollama'],
                        help='LLM类型 (覆盖配置文件，选项: local, openai, ollama)')

    parser.add_argument('--ollama-model',
                        type=str,
                        help='Ollama模型名称 (例如: llama2, codellama, phi3)')

    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = get_config(args.config)
    else:
        config = get_config()

    # 命令行参数覆盖配置文件
    if args.llm_type:
        config.llm_type = args.llm_type
        logger.info(f"使用命令行指定的LLM类型: {args.llm_type}")

    if args.ollama_model:
        config.ollama_model = args.ollama_model
        logger.info(f"使用命令行指定的Ollama模型: {args.ollama_model}")

    if not config.validate():
        print("[x] 配置验证失败")
        sys.exit(1)

    # 执行命令
    try:
        if args.stats:
            show_stats()

        elif args.dataset:
            # 处理完整流程
            if args.full_pipeline or args.dataset == 'all':
                # 使用 --source-dataset 指定的数据源，默认 refinement (有任务四标签)
                task_type = args.source_dataset if hasattr(
                    args, 'source_dataset') else 'refinement'
                review_dataset_full_pipeline(task_type, args.limit, args.split)
            else:
                # 单任务执行
                if args.dataset == 'task1':
                    task_type = 'quality'
                elif args.dataset == 'task2':
                    task_type = 'comment'  # Task2 使用 comment 数据
                elif args.dataset == 'task3':
                    task_type = 'comment'
                elif args.dataset == 'task4':
                    task_type = 'refinement'
                else:
                    task_type = 'quality'

                review_dataset(task_type, args.limit, args.split)

        elif args.file:
            review_file(args.file, args.language)

        elif args.code:
            review_code(args.code, args.language, args.diff)

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)

    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        print(f"\n[x] 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
