"""
代码分析工具
提供静态代码分析、复杂度计算等功能
"""

import ast
import re
from typing import Dict, List, Any, Optional
from radon.complexity import cc_visit
from radon.metrics import mi_visit
import pylint.lint
from io import StringIO
import sys
import logging

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """代码分析器"""

    @staticmethod
    def analyze_complexity(code: str,
                           language: str = "python") -> Dict[str, Any]:
        """
        分析代码复杂度
        
        Args:
            code: 代码字符串
            language: 编程语言
            
        Returns:
            复杂度分析结果
        """
        if language.lower() not in ["python", "py"]:
            return {"error": f"暂不支持{language}语言的复杂度分析"}

        try:
            # 圈复杂度
            complexity_results = cc_visit(code)

            results = {
                "total_complexity":
                sum(c.complexity for c in complexity_results),
                "average_complexity":
                sum(c.complexity for c in complexity_results) /
                len(complexity_results) if complexity_results else 0,
                "max_complexity":
                max((c.complexity for c in complexity_results), default=0),
                "functions": [{
                    "name": c.name,
                    "complexity": c.complexity,
                    "line": c.lineno,
                    "rank": c.rank
                } for c in complexity_results]
            }

            # 可维护性指数
            try:
                mi_score = mi_visit(code, multi=True)
                results["maintainability_index"] = mi_score
            except:
                results["maintainability_index"] = None

            return results

        except Exception as e:
            logger.error(f"复杂度分析失败: {e}")
            return {"error": str(e)}

    @staticmethod
    def analyze_ast(code: str, language: str = "python") -> Dict[str, Any]:
        """
        分析抽象语法树
        
        Args:
            code: 代码字符串
            language: 编程语言
            
        Returns:
            AST分析结果
        """
        if language.lower() not in ["python", "py"]:
            return {"error": f"暂不支持{language}语言的AST分析"}

        try:
            tree = ast.parse(code)

            stats = {
                "num_functions": 0,
                "num_classes": 0,
                "num_imports": 0,
                "num_loops": 0,
                "num_conditions": 0,
                "num_try_except": 0
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    stats["num_functions"] += 1
                elif isinstance(node, ast.ClassDef):
                    stats["num_classes"] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    stats["num_imports"] += 1
                elif isinstance(node, (ast.For, ast.While)):
                    stats["num_loops"] += 1
                elif isinstance(node, ast.If):
                    stats["num_conditions"] += 1
                elif isinstance(node, ast.Try):
                    stats["num_try_except"] += 1

            return stats

        except Exception as e:
            logger.error(f"AST分析失败: {e}")
            return {"error": str(e)}

    @staticmethod
    def run_pylint(code: str) -> Dict[str, Any]:
        """
        运行pylint静态分析
        
        Args:
            code: 代码字符串
            
        Returns:
            pylint分析结果
        """
        try:
            # 将代码写入临时变量
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w',
                                             suffix='.py',
                                             delete=False) as f:
                f.write(code)
                temp_file = f.name

            # 捕获pylint输出
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                pylint.lint.Run([temp_file, '--output-format=json'],
                                exit=False)
                output = sys.stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            # 清理临时文件
            import os
            os.unlink(temp_file)

            # 解析JSON输出
            import json
            try:
                results = json.loads(output) if output else []
                return {
                    "issues_count": len(results),
                    "issues": results[:10]  # 只返回前10个问题
                }
            except:
                return {"error": "无法解析pylint输出"}

        except Exception as e:
            logger.error(f"pylint分析失败: {e}")
            return {"error": str(e)}

    @staticmethod
    def extract_code_features(code: str,
                              language: str = "python") -> Dict[str, Any]:
        """
        提取代码特征（用于Agent决策）
        
        Args:
            code: 代码字符串
            language: 编程语言
            
        Returns:
            代码特征字典
        """
        features = {
            "language": language,
            "length": len(code),
            "num_lines": code.count('\n') + 1,
            "has_comments": bool(re.search(r'#.*|""".*?"""', code, re.DOTALL)),
            "has_docstring": bool(re.search(r'""".*?"""', code, re.DOTALL))
        }

        # 添加复杂度特征
        if language.lower() in ["python", "py"]:
            complexity = CodeAnalyzer.analyze_complexity(code, language)
            if "error" not in complexity:
                features["complexity"] = complexity.get(
                    "average_complexity", 0)
                features["max_complexity"] = complexity.get(
                    "max_complexity", 0)

            # 添加AST特征
            ast_stats = CodeAnalyzer.analyze_ast(code, language)
            if "error" not in ast_stats:
                features.update(ast_stats)

        return features

    @staticmethod
    def analyze_diff(old_code: str,
                     new_code: str,
                     language: str = "python") -> Dict[str, Any]:
        """
        分析代码差异
        
        Args:
            old_code: 旧代码
            new_code: 新代码
            language: 编程语言
            
        Returns:
            差异分析结果
        """
        try:
            old_features = CodeAnalyzer.extract_code_features(
                old_code, language)
            new_features = CodeAnalyzer.extract_code_features(
                new_code, language)

            diff_analysis = {
                "lines_changed":
                abs(new_features["num_lines"] - old_features["num_lines"]),
                "complexity_change":
                new_features.get("complexity", 0) -
                old_features.get("complexity", 0),
                "functions_added":
                new_features.get("num_functions", 0) -
                old_features.get("num_functions", 0),
                "old_features":
                old_features,
                "new_features":
                new_features
            }

            return diff_analysis

        except Exception as e:
            logger.error(f"差异分析失败: {e}")
            return {"error": str(e)}


# LangChain工具包装
def create_code_analysis_tools():
    """创建代码分析工具列表（用于LangChain）"""
    from langchain.tools import Tool

    tools = [
        Tool(name="analyze_complexity",
             func=lambda code: CodeAnalyzer.analyze_complexity(code),
             description="分析Python代码的圈复杂度和可维护性指数"),
        Tool(name="analyze_ast",
             func=lambda code: CodeAnalyzer.analyze_ast(code),
             description="分析Python代码的抽象语法树，统计函数、类、循环等"),
        Tool(name="extract_features",
             func=lambda code: CodeAnalyzer.extract_code_features(code),
             description="提取代码特征，用于决策")
    ]

    return tools
    # 测试代码分析
    test_code = """
def fibonacci(n):
    '''计算斐波那契数列'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
"""

    analyzer = CodeAnalyzer()

    print("=== 复杂度分析 ===")
    print(analyzer.analyze_complexity(test_code))

    print("\n=== AST分析 ===")
    print(analyzer.analyze_ast(test_code))

    print("\n=== 代码特征 ===")
    print(analyzer.extract_code_features(test_code))
