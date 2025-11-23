"""
Level 2 Task 1: 代码质量评估 - 推理脚本

使用 LLM 和提示工程进行代码质量评估
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from level2.shared.llm_client import create_llm_client
from level2.shared.prompt_utils import build_quality_estimation_prompt
from config import DIFF_QUALITY_DIR, LEVEL2_OUTPUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_data(data_file):
    """加载测试数据"""
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def run_inference(args):
    """运行推理"""
    # 创建 LLM 客户端
    logger.info(f"Creating LLM client: {args.provider}/{args.model}")
    client = create_llm_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    # 加载测试数据
    data_file = DIFF_QUALITY_DIR / "cls-test.jsonl"
    logger.info(f"Loading test data from {data_file}")
    test_data = load_test_data(data_file)

    if args.max_samples:
        test_data = test_data[: args.max_samples]

    logger.info(f"Processing {len(test_data)} samples")

    # 提示词文件路径
    prompt_dir = Path(__file__).parent / "prompts"
    system_prompt_file = prompt_dir / "system_prompt.txt"
    task_prompt_file = prompt_dir / "task_prompt.txt"

    # 推理
    predictions = []
    failed_samples = []

    for idx, sample in enumerate(tqdm(test_data, desc="Inference")):
        try:
            # 构建提示词
            old_code = sample.get("oldf", "")
            diff_code = sample.get("old_hunk", "")
            language = sample.get("lang", "code")

            messages = build_quality_estimation_prompt(
                old_code=old_code,
                diff_code=diff_code,
                language=language,
                system_prompt_file=str(system_prompt_file),
                task_prompt_file=str(task_prompt_file),
            )

            # 调用 LLM
            response = client.chat_completion(
                messages=messages,
                temperature=args.temperature,
                retry_attempts=args.retry_attempts,
            )

            # 解析响应
            result = client.extract_json(response)

            if result and "needs_review" in result:
                pred_label = result["needs_review"]
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
            else:
                # 解析失败,使用默认值
                pred_label = 1  # 保守策略:默认需要评审
                confidence = 0.5
                reasoning = "Failed to parse LLM response"
                logger.warning(
                    f"Failed to parse response for sample {idx}: {response[:200]}"
                )

            predictions.append(
                {
                    "idx": idx,
                    "sample_id": sample.get("ids", [None])[0],
                    "prediction": pred_label,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "ground_truth": sample.get("label", -1),
                    "raw_response": response,
                }
            )

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            failed_samples.append(idx)
            predictions.append(
                {
                    "idx": idx,
                    "sample_id": sample.get("ids", [None])[0],
                    "prediction": 1,  # 默认需要评审
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}",
                    "ground_truth": sample.get("label", -1),
                    "raw_response": "",
                }
            )

    # 保存结果
    output_dir = LEVEL2_OUTPUT / "task1"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / args.output_file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    logger.info(f"Predictions saved to {output_file}")
    logger.info(f"Total samples: {len(test_data)}")
    logger.info(f"Failed samples: {len(failed_samples)}")

    return predictions


def main():
    parser = argparse.ArgumentParser()

    # LLM 参数
    parser.add_argument(
        "--provider", type=str, default="openai", choices=["openai", "anthropic"]
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)

    # 推理参数
    parser.add_argument(
        "--max_samples", type=int, default=None, help="限制样本数量(用于测试)"
    )
    parser.add_argument("--retry_attempts", type=int, default=3)
    parser.add_argument("--output_file", type=str, default="predictions.json")

    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
