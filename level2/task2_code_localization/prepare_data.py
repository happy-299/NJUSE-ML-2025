"""
Level 2 Task 2: 问题代码定位 - 数据准备脚本

从原始数据中提取和准备问题定位任务的数据
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config import COMMENT_GEN_DIR, CODE_REFINE_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_diff(diff_text):
    """解析 diff 文本,提取修改的行号"""
    added_lines = []
    deleted_lines = []

    # 解析 @@ -start,count +start,count @@ 格式
    header_pattern = r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@"

    lines = diff_text.split("\n")
    current_new_line = 0

    for line in lines:
        # 解析 header
        match = re.match(header_pattern, line)
        if match:
            current_new_line = int(match.group(3)) - 1  # 0-based
            continue

        # 解析内容
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(current_new_line)
            current_new_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            # 删除的行不计入新代码行号
            pass
        elif line.startswith(" "):
            current_new_line += 1

    return added_lines, deleted_lines


def add_line_numbers(code):
    """给代码添加行号"""
    lines = code.split("\n")
    numbered_lines = [f"{i:4d} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def extract_new_code_from_diff(old_code, diff):
    """从旧代码和 diff 提取新代码(简化版)"""
    # 这是一个简化实现,实际应该使用 diff 库
    # 这里假设 diff 中的 '+' 行是新增的
    lines = diff.split("\n")
    new_lines = []

    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            new_lines.append(line[1:])  # 去掉 '+'
        elif line.startswith(" "):
            new_lines.append(line[1:])  # 去掉 ' '

    return "\n".join(new_lines) if new_lines else old_code


def prepare_localization_data(source_dir, output_file, max_samples=None):
    """准备问题定位数据"""
    logger.info(f"Preparing localization data from {source_dir}")

    # 收集数据
    data = []
    file_patterns = ["*-test.jsonl", "*-valid.jsonl"]

    files = []
    for pattern in file_patterns:
        files.extend(source_dir.glob(pattern))

    for data_file in files:
        logger.info(f"Processing {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())

                # 提取字段
                old_code = sample.get("oldf", "")
                diff = sample.get("old_hunk", "") or sample.get("hunk", "")
                comment = sample.get("comment", "")
                lang = sample.get("lang", "code")

                if not old_code or not diff or not comment:
                    continue

                # 解析 diff,获取修改的行号
                added_lines, _ = parse_diff(diff)

                if not added_lines:
                    # 如果没有新增行,跳过
                    continue

                # 提取新代码
                new_code = extract_new_code_from_diff(old_code, diff)

                # 添加行号
                old_code_numbered = add_line_numbers(old_code)
                new_code_numbered = add_line_numbers(new_code)

                data.append(
                    {
                        "sample_id": sample.get("ids", [None])[0],
                        "old_code": old_code,
                        "old_code_numbered": old_code_numbered,
                        "new_code": new_code,
                        "new_code_numbered": new_code_numbered,
                        "diff": diff,
                        "comment": comment,
                        "language": lang,
                        "ground_truth_lines": added_lines,  # 简化:假设所有新增行都需要修改
                        "repo": sample.get("repo", ""),
                    }
                )

                if max_samples and len(data) >= max_samples:
                    break

        if max_samples and len(data) >= max_samples:
            break

    logger.info(f"Prepared {len(data)} samples")

    # 保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Data saved to {output_file}")

    return data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="comment_generation",
        choices=["comment_generation", "code_refinement"],
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_file", type=str, default="localization_data.json")

    args = parser.parse_args()

    # 选择数据源
    if args.source == "comment_generation":
        source_dir = COMMENT_GEN_DIR
    else:
        source_dir = CODE_REFINE_DIR

    # 输出文件
    output_file = PROCESSED_DATA_DIR / args.output_file

    # 准备数据
    prepare_localization_data(source_dir, output_file, args.max_samples)


if __name__ == "__main__":
    main()
