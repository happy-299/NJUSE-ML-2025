"""
Level 2 Task 2: 问题代码定位 - 推理脚本

使用 LLM 和提示工程进行问题代码定位
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
from level2.shared.prompt_utils import load_prompt, format_prompt, create_messages
from config import PROCESSED_DATA_DIR, LEVEL2_OUTPUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_file):
    """加载数据"""
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_localization_prompt(sample, system_prompt_file, task_prompt_file):
    """构建定位提示词"""
    # 加载模板
    system_prompt = load_prompt(system_prompt_file)
    task_template = load_prompt(task_prompt_file)

    # 格式化
    user_prompt = format_prompt(
        task_template,
        language=sample["language"],
        old_code_numbered=sample["old_code_numbered"],
        new_code_numbered=sample["new_code_numbered"],
        diff_code=sample["diff"],
        comment=sample["comment"],
    )

    return create_messages(system_prompt, user_prompt)


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

    # 加载数据
    data_file = PROCESSED_DATA_DIR / args.data_file
    logger.info(f"Loading data from {data_file}")
    data = load_data(data_file)

    if args.max_samples:
        data = data[: args.max_samples]

    logger.info(f"Processing {len(data)} samples")

    # 提示词文件
    prompt_dir = Path(__file__).parent / "prompts"
    system_prompt_file = str(prompt_dir / "system_prompt.txt")
    task_prompt_file = str(prompt_dir / "task_prompt.txt")

    # 推理
    predictions = []
    failed_samples = []

    for idx, sample in enumerate(tqdm(data, desc="Inference")):
        try:
            # 构建提示词
            messages = build_localization_prompt(
                sample, system_prompt_file, task_prompt_file
            )

            # 调用 LLM
            response = client.chat_completion(
                messages=messages,
                temperature=args.temperature,
                retry_attempts=args.retry_attempts,
            )

            # 解析响应
            result = client.extract_json(response)

            if result and "line_indices" in result:
                line_indices = result["line_indices"]
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
            else:
                # 解析失败,使用默认值
                line_indices = sample.get("ground_truth_lines", [])[:1]  # 猜测第一行
                confidence = 0.0
                reasoning = "Failed to parse LLM response"
                logger.warning(f"Failed to parse response for sample {idx}")

            predictions.append(
                {
                    "idx": idx,
                    "sample_id": sample.get("sample_id"),
                    "prediction": line_indices,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "ground_truth": sample.get("ground_truth_lines", []),
                    "raw_response": response,
                }
            )

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            failed_samples.append(idx)
            predictions.append(
                {
                    "idx": idx,
                    "sample_id": sample.get("sample_id"),
                    "prediction": [],
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}",
                    "ground_truth": sample.get("ground_truth_lines", []),
                    "raw_response": "",
                }
            )

    # 保存结果
    output_dir = LEVEL2_OUTPUT / "task2"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / args.output_file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    logger.info(f"Predictions saved to {output_file}")
    logger.info(f"Total samples: {len(data)}")
    logger.info(f"Failed samples: {len(failed_samples)}")

    return predictions


def main():
    parser = argparse.ArgumentParser()

    # LLM 参数
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)

    # 数据参数
    parser.add_argument("--data_file", type=str, default="localization_data.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--retry_attempts", type=int, default=3)
    parser.add_argument("--output_file", type=str, default="predictions.json")

    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
