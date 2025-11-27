"""
代码分析工具
"""

import re
from typing import Dict, Any


def analyze_code_complexity(code: str) -> Dict[str, Any]:
    """
    分析代码复杂度
    
    Args:
        code: 源代码
        
    Returns:
        复杂度指标字典
    """
    if not code:
        return {
            "lines": 0,
            "functions": 0,
            "classes": 0,
            "complexity_score": 0,
            "nested_depth": 0
        }
    
    lines = code.split('\n')
    
    # 统计行数
    total_lines = len(lines)
    code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
    blank_lines = len([l for l in lines if not l.strip()])
    comment_lines = len([l for l in lines if l.strip().startswith('#')])
    
    # 统计函数和类
    function_count = len(re.findall(r'\bdef\s+\w+', code))
    class_count = len(re.findall(r'\bclass\s+\w+', code))
    
    # 计算嵌套深度
    max_indent = 0
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent // 4)
    
    # 计算复杂度分数 (简单启发式)
    complexity_score = (
        code_lines * 0.1 +
        function_count * 2 +
        class_count * 3 +
        max_indent * 1.5
    )
    
    return {
        "total_lines": total_lines,
        "code_lines": code_lines,
        "blank_lines": blank_lines,
        "comment_lines": comment_lines,
        "functions": function_count,
        "classes": class_count,
        "max_nested_depth": max_indent,
        "complexity_score": round(complexity_score, 2)
    }


def count_code_lines(code: str) -> int:
    """统计代码行数"""
    if not code:
        return 0
    return len([l for l in code.split('\n') if l.strip()])


def analyze_diff(diff: str) -> Dict[str, Any]:
    """
    分析 diff 内容
    
    Args:
        diff: diff 文本
        
    Returns:
        diff 分析结果
    """
    if not diff:
        return {
            "additions": 0,
            "deletions": 0,
            "changes": 0
        }
    
    lines = diff.split('\n')
    additions = len([l for l in lines if l.startswith('+') and not l.startswith('+++')])
    deletions = len([l for l in lines if l.startswith('-') and not l.startswith('---')])
    
    return {
        "additions": additions,
        "deletions": deletions,
        "changes": additions + deletions,
        "net_change": additions - deletions
    }


def extract_changed_lines(diff: str) -> Dict[str, list]:
    """
    从 diff 中提取变更的行号
    
    Args:
        diff: diff 文本
        
    Returns:
        包含 added_lines 和 deleted_lines 的字典
    """
    added_lines = []
    deleted_lines = []
    
    current_old_line = 0
    current_new_line = 0
    
    for line in diff.split('\n'):
        # 解析 hunk header: @@ -start,count +start,count @@
        hunk_match = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', line)
        if hunk_match:
            current_old_line = int(hunk_match.group(1))
            current_new_line = int(hunk_match.group(2))
            continue
        
        if line.startswith('+') and not line.startswith('+++'):
            added_lines.append(current_new_line)
            current_new_line += 1
        elif line.startswith('-') and not line.startswith('---'):
            deleted_lines.append(current_old_line)
            current_old_line += 1
        else:
            current_old_line += 1
            current_new_line += 1
    
    return {
        "added_lines": added_lines,
        "deleted_lines": deleted_lines
    }
