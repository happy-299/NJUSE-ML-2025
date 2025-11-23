"""
Level 2 共享模块: 提示词工具函数
"""

import os
from pathlib import Path
from typing import Dict, Optional


def load_prompt(prompt_file: str) -> str:
    """
    加载提示词文件

    Args:
        prompt_file: 提示词文件路径

    Returns:
        提示词文本
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


def format_prompt(template: str, **kwargs) -> str:
    """
    格式化提示词模板

    Args:
        template: 提示词模板
        **kwargs: 替换参数

    Returns:
        格式化后的提示词
    """
    return template.format(**kwargs)


def create_messages(
    system_prompt: str, user_prompt: str, assistant_prompt: Optional[str] = None
) -> list:
    """
    创建消息列表

    Args:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        assistant_prompt: 助手提示词(可选,用于 few-shot)

    Returns:
        消息列表
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if assistant_prompt:
        messages.append({"role": "assistant", "content": assistant_prompt})

    return messages


def truncate_code(code: str, max_length: int = 5000) -> str:
    """
    截断过长的代码

    Args:
        code: 代码文本
        max_length: 最大长度

    Returns:
        截断后的代码
    """
    if len(code) <= max_length:
        return code

    # 尝试保留前后部分
    half = max_length // 2
    return code[:half] + "\n\n... (truncated) ...\n\n" + code[-half:]


def extract_language_from_filename(filename: str) -> str:
    """
    从文件名提取编程语言

    Args:
        filename: 文件名

    Returns:
        编程语言名称
    """
    ext_to_lang = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
    }

    ext = Path(filename).suffix.lower() if filename else ""
    return ext_to_lang.get(ext, "code")


def build_quality_estimation_prompt(
    old_code: str,
    diff_code: str,
    language: str = "code",
    system_prompt_file: Optional[str] = None,
    task_prompt_file: Optional[str] = None,
) -> list:
    """
    构建代码质量评估的提示词

    Args:
        old_code: 旧代码
        diff_code: 代码差异
        language: 编程语言
        system_prompt_file: 系统提示词文件路径
        task_prompt_file: 任务提示词文件路径

    Returns:
        消息列表
    """
    # 加载提示词模板
    if system_prompt_file and os.path.exists(system_prompt_file):
        system_prompt = load_prompt(system_prompt_file)
    else:
        system_prompt = "You are an expert code reviewer."

    if task_prompt_file and os.path.exists(task_prompt_file):
        task_template = load_prompt(task_prompt_file)
    else:
        task_template = """
# Code Change Analysis

## Old File Content
```{language}
{old_code}
```

## Code Diff
```diff
{diff_code}
```

Analyze this code change and determine if it needs human review.
"""

    # 截断过长的代码
    old_code = truncate_code(old_code, max_length=4000)
    diff_code = truncate_code(diff_code, max_length=2000)

    # 格式化用户提示词
    user_prompt = format_prompt(
        task_template, language=language, old_code=old_code, diff_code=diff_code
    )

    return create_messages(system_prompt, user_prompt)


def build_localization_prompt(
    old_code: str,
    diff_code: str,
    comment: str,
    language: str = "code",
    system_prompt_file: Optional[str] = None,
    task_prompt_file: Optional[str] = None,
) -> list:
    """
    构建问题定位的提示词

    Args:
        old_code: 旧代码
        diff_code: 代码差异
        comment: 评审意见
        language: 编程语言
        system_prompt_file: 系统提示词文件路径
        task_prompt_file: 任务提示词文件路径

    Returns:
        消息列表
    """
    # 加载提示词模板
    if system_prompt_file and os.path.exists(system_prompt_file):
        system_prompt = load_prompt(system_prompt_file)
    else:
        system_prompt = (
            "You are an expert code reviewer specializing in locating code issues."
        )

    if task_prompt_file and os.path.exists(task_prompt_file):
        task_template = load_prompt(task_prompt_file)
    else:
        task_template = """
# Code Issue Localization

## Old File Content
```{language}
{old_code}
```

## Code Diff
```diff
{diff_code}
```

## Review Comment
{comment}

Your task: Locate the specific lines in the code change that need to be modified based on the review comment.
Output the line indices (0-based) in JSON format: {{"line_indices": [...]}}
"""

    # 截断过长的代码
    old_code = truncate_code(old_code, max_length=4000)
    diff_code = truncate_code(diff_code, max_length=2000)

    # 格式化用户提示词
    user_prompt = format_prompt(
        task_template,
        language=language,
        old_code=old_code,
        diff_code=diff_code,
        comment=comment,
    )

    return create_messages(system_prompt, user_prompt)


if __name__ == "__main__":
    # 测试
    old_code = "def foo():\n    return 1"
    diff_code = "@@ -1,1 +1,1 @@\n-    return 1\n+    return 2"

    messages = build_quality_estimation_prompt(old_code, diff_code, "python")

    print("System Prompt:")
    print(messages[0]["content"])
    print("\nUser Prompt:")
    print(messages[1]["content"])
