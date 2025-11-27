"""
Level 2 Task 2: Code Localization - Inference Script

Using LLM and prompt engineering for code issue localization
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm

# Add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from level2.shared.llm_client import create_llm_client
from level2.shared.prompt_utils import load_prompt, format_prompt, create_messages
from config import PROCESSED_DATA_DIR, LEVEL2_OUTPUT, COMMENT_GEN_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_file):
    """Load data"""
    with open(data_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_data(data_file, max_samples=None):
    """Load raw jsonl data directly"""
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            data.append(sample)
            if max_samples and len(data) >= max_samples:
                break
    return data


def add_line_numbers(code):
    """Add line numbers to code"""
    lines = code.split("\n")
    numbered_lines = [f"{i:4d} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def build_localization_prompt(sample, system_prompt_file, task_prompt_file):
    """Build localization prompt"""
    # Load templates
    system_prompt = load_prompt(system_prompt_file)
    task_template = load_prompt(task_prompt_file)

    # Get fields from raw data
    old_code = sample.get("oldf", "")
    new_code = sample.get("new", "") or sample.get("hunk", "")
    diff_code = (
        sample.get("patch", "") or sample.get("old_hunk", "") or sample.get("hunk", "")
    )
    comment = sample.get("comment", "") or sample.get("msg", "")
    language = sample.get("lang", "code")

    # Add line numbers
    old_code_numbered = add_line_numbers(old_code) if old_code else ""
    new_code_numbered = add_line_numbers(new_code) if new_code else ""

    # Format
    user_prompt = format_prompt(
        task_template,
        language=language,
        old_code_numbered=old_code_numbered,
        new_code_numbered=new_code_numbered,
        diff_code=diff_code,
        comment=comment,
    )

    return create_messages(system_prompt, user_prompt)


def run_inference(args):
    """Run inference"""
    # Create LLM client
    logger.info(f"Creating LLM client: {args.provider}/{args.model}")
    client = create_llm_client(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
    )

    # Load data - try processed data first, fallback to raw data
    data_file = PROCESSED_DATA_DIR / args.data_file
    if data_file.exists():
        logger.info(f"Loading processed data from {data_file}")
        data = load_data(data_file)
    else:
        # Use raw data from Comment_Generation
        raw_file = COMMENT_GEN_DIR / "msg-test.jsonl"
        logger.info(f"Loading raw data from {raw_file}")
        data = load_raw_data(raw_file, args.max_samples)

    if args.max_samples and len(data) > args.max_samples:
        data = data[: args.max_samples]

    logger.info(f"Processing {len(data)} samples")

    # Prompt files
    prompt_dir = Path(__file__).parent / "prompts"
    system_prompt_file = str(prompt_dir / "system_prompt.txt")
    task_prompt_file = str(prompt_dir / "task_prompt.txt")

    # Inference
    predictions = []
    failed_samples = []

    for idx, sample in enumerate(tqdm(data, desc="Inference")):
        try:
            # Build prompt
            messages = build_localization_prompt(
                sample, system_prompt_file, task_prompt_file
            )

            # Call LLM
            response = client.chat_completion(
                messages=messages,
                temperature=args.temperature,
                retry_attempts=args.retry_attempts,
            )

            # Parse response
            result = client.extract_json(response)

            if result and "line_indices" in result:
                line_indices = result["line_indices"]
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
            else:
                # Parse failed, use default
                line_indices = [0]  # Guess first line
                confidence = 0.0
                reasoning = "Failed to parse LLM response"
                logger.warning(f"Failed to parse response for sample {idx}")

            predictions.append(
                {
                    "idx": idx,
                    "sample_id": sample.get("id", sample.get("idx", idx)),
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
                    "sample_id": sample.get("id", sample.get("idx", idx)),
                    "prediction": [],
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}",
                    "ground_truth": sample.get("ground_truth_lines", []),
                    "raw_response": "",
                }
            )

    # Save results
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

    # LLM parameters
    parser.add_argument("--provider", type=str, default="deepseek")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Data parameters
    parser.add_argument("--data_file", type=str, default="localization_data.json")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--retry_attempts", type=int, default=3)
    parser.add_argument("--output_file", type=str, default="predictions.json")

    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
