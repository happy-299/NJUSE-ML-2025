"""
简单提示词模板 - 待团队完整版本替换
"""

# Code Refinement - 代码优化任务
REFINEMENT_DETECTION_PROMPT = """Review the following code and identify any issues:

Code:
```{language}
{code}
```

List any bugs, security issues, or bad practices you find."""

REFINEMENT_SUGGESTION_PROMPT = """Based on the issues found, provide specific improvement suggestions:

Issues:
{issues}

Give clear, actionable suggestions."""

# Comment Generation - 注释生成任务
COMMENT_GENERATION_PROMPT = """Generate a code review comment for this change:

Code Change:
```
{diff}
```

Write a concise review comment."""

# Diff Quality Estimation - 质量评估任务
QUALITY_ESTIMATION_PROMPT = """Evaluate the quality of this code change (good/bad):

Change:
```
{diff}
```

Is this a good or bad change? Explain why."""

# 通用检测提示词
GENERAL_DETECTION_PROMPT = """You are a code reviewer. Review this {language} code:

```{language}
{code}
```

{diff_section}

Identify potential issues in these categories:
1. Bugs or errors
2. Security problems  
3. Performance issues
4. Code style problems
5. Maintainability concerns

List each issue on a new line."""

# 通用建议提示词
GENERAL_SUGGESTION_PROMPT = """Based on these code issues:

{issues}

Provide specific improvement suggestions. Each suggestion should be:
- Clear and actionable
- Start with a bullet point (•)
- One per line"""


def get_detection_prompt(task_type="refinement"):
    """获取检测提示词"""
    prompts = {
        "refinement": REFINEMENT_DETECTION_PROMPT,
        "comment": COMMENT_GENERATION_PROMPT,
        "quality": QUALITY_ESTIMATION_PROMPT,
        "general": GENERAL_DETECTION_PROMPT
    }
    return prompts.get(task_type, GENERAL_DETECTION_PROMPT)


def get_suggestion_prompt(task_type="refinement"):
    """获取建议提示词"""
    prompts = {
        "refinement": REFINEMENT_SUGGESTION_PROMPT,
        "general": GENERAL_SUGGESTION_PROMPT
    }
    return prompts.get(task_type, GENERAL_SUGGESTION_PROMPT)
