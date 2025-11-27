"""
代码审查工作流
使用LangGraph实现多阶段代码审查流程
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import logging

from ..agents.state import AgentState, ReviewStage, ReviewPriority, create_initial_state
from ..tools.code_analyzer import CodeAnalyzer
from ..llm_factory import get_llm
from ..config import get_config
from ..prompts import get_detection_prompt, get_suggestion_prompt

logger = logging.getLogger(__name__)


class CodeReviewWorkflow:
    """代码审查工作流"""

    def __init__(self, config=None):
        """
        初始化工作流
        
        Args:
            config: 配置对象
        """
        self.config = config or get_config()
        self.llm = get_llm(self.config)
        self.analyzer = CodeAnalyzer()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建LangGraph状态图"""

        # 创建状态图
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("analyze", self.analyze_code)
        workflow.add_node("detect_issues", self.detect_issues)
        workflow.add_node("generate_suggestions", self.generate_suggestions)
        workflow.add_node("finalize", self.finalize_review)

        # 设置入口点
        workflow.set_entry_point("analyze")

        # 添加边（状态转换）
        workflow.add_edge("analyze", "detect_issues")
        workflow.add_edge("detect_issues", "generate_suggestions")
        workflow.add_edge("generate_suggestions", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def analyze_code(self, state: AgentState) -> AgentState:
        """
        分析代码阶段
        提取代码特征和复杂度信息
        """
        logger.info("=== 阶段1: 代码分析 ===")

        try:
            code = state["code"]
            language = state["language"]

            # 提取代码特征
            features = self.analyzer.extract_code_features(code, language)
            state["code_features"] = features

            # 分析复杂度
            if language.lower() in ["python", "py"]:
                complexity = self.analyzer.analyze_complexity(code, language)
                state["complexity_analysis"] = complexity

                # 判断是否需要详细审查
                avg_complexity = complexity.get("average_complexity", 0)
                if avg_complexity > 10:
                    state["needs_detailed_review"] = True
                    state["review_priority"] = ReviewPriority.HIGH
                elif avg_complexity > 5:
                    state["review_priority"] = ReviewPriority.MEDIUM
                else:
                    state["review_priority"] = ReviewPriority.LOW

            state["current_stage"] = ReviewStage.ANALYZE

            logger.info(f"代码特征: {features}")
            logger.info(f"审查优先级: {state['review_priority']}")

        except Exception as e:
            logger.error(f"代码分析失败: {e}")
            state["error"] = str(e)

        return state

    def detect_issues(self, state: AgentState) -> AgentState:
        """
        检测问题阶段
        使用LLM检测代码中的潜在问题
        """
        logger.info("=== 阶段2: 问题检测 ===")

        try:
            code = state["code"]
            diff = state.get("diff", "")
            language = state["language"]
            features = state.get("code_features", {})

            # 构建提示词
            prompt = self._build_detection_prompt(code, diff, language,
                                                  features)

            # 调用LLM检测问题
            logger.info("调用LLM进行问题检测...")
            response = self.llm.invoke(prompt)

            # 解析响应
            issues = self._parse_issues(response)
            state["issues_found"] = issues

            state["current_stage"] = ReviewStage.DETECT

            logger.info(f"发现 {len(issues)} 个问题")

        except Exception as e:
            logger.error(f"问题检测失败: {e}")
            state["error"] = str(e)

        return state

    def generate_suggestions(self, state: AgentState) -> AgentState:
        """
        生成建议阶段
        为发现的问题生成改进建议
        """
        logger.info("=== 阶段3: 生成建议 ===")

        try:
            code = state["code"]
            issues = state.get("issues_found", [])
            language = state["language"]

            if not issues:
                logger.info("未发现问题，跳过建议生成")
                state["suggestions"] = ["代码质量良好，未发现明显问题"]
                state["current_stage"] = ReviewStage.SUGGEST
                return state

            # 构建提示词
            prompt = self._build_suggestion_prompt(code, issues, language)

            # 调用LLM生成建议
            logger.info("调用LLM生成改进建议...")
            response = self.llm.invoke(prompt)

            # 解析建议
            suggestions = self._parse_suggestions(response)
            state["suggestions"] = suggestions

            state["current_stage"] = ReviewStage.SUGGEST

            logger.info(f"生成了 {len(suggestions)} 条建议")

        except Exception as e:
            logger.error(f"建议生成失败: {e}")
            state["error"] = str(e)

        return state

    def finalize_review(self, state: AgentState) -> AgentState:
        """
        最终整合阶段
        生成完整的审查报告
        """
        logger.info("=== 阶段4: 最终整合 ===")

        try:
            issues = state.get("issues_found", [])
            suggestions = state.get("suggestions", [])
            features = state.get("code_features", {})

            # 计算质量评分
            quality_score = self._calculate_quality_score(issues, features)
            state["quality_score"] = quality_score

            # 生成审查评论
            review_comment = self._generate_review_comment(
                issues, suggestions, quality_score)
            state["review_comment"] = review_comment

            state["current_stage"] = ReviewStage.FINALIZE
            state["completed"] = True

            logger.info(f"审查完成，质量评分: {quality_score:.2f}")

        except Exception as e:
            logger.error(f"最终整合失败: {e}")
            state["error"] = str(e)

        return state

    def _build_detection_prompt(self, code: str, diff: str, language: str,
                                features: Dict[str, Any]) -> str:
        """构建问题检测提示词"""

        # 使用简单提示词模板
        task_type = "general"
        template = get_detection_prompt(task_type)

        diff_section = ""
        if diff:
            diff_section = f"""
Code Changes (Diff):
```
{diff}
```
"""

        prompt = template.format(language=language,
                                 code=code,
                                 diff_section=diff_section)

        # 添加度量信息
        prompt += f"""

Code Metrics:
- Lines: {features.get('num_lines', 'N/A')}
- Complexity: {features.get('complexity', 'N/A')}
- Functions: {features.get('num_functions', 'N/A')}
"""

        return prompt

    def _build_suggestion_prompt(self, code: str, issues: list,
                                 language: str) -> str:
        """构建建议生成提示词"""

        # 使用简单提示词模板
        task_type = "general"
        template = get_suggestion_prompt(task_type)

        issues_text = "\n".join(f"- {issue.get('description', str(issue))}"
                                for issue in issues)

        prompt = template.format(issues=issues_text)

        return prompt

    def _parse_issues(self, response: str) -> list:
        """解析问题检测响应"""
        # 简单解析：每行一个问题
        lines = response.strip().split('\n')
        issues = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                issues.append({
                    "description": line.lstrip('- •*'),
                    "severity": "medium"  # 默认中等严重性
                })

        return issues[:10]  # 限制最多10个问题

    def _parse_suggestions(self, response: str) -> list:
        """解析建议响应"""
        lines = response.strip().split('\n')
        suggestions = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                suggestions.append(line.lstrip('- •*'))

        return suggestions[:10]  # 限制最多10条建议

    def _calculate_quality_score(self, issues: list,
                                 features: Dict[str, Any]) -> float:
        """计算代码质量评分（0-100）"""
        base_score = 100.0

        # 根据问题数量扣分
        issue_penalty = len(issues) * 10
        base_score -= min(issue_penalty, 50)

        # 根据复杂度调整
        complexity = features.get("complexity", 0)
        if complexity > 15:
            base_score -= 10
        elif complexity > 10:
            base_score -= 5

        return max(0, min(100, base_score))

    def _generate_review_comment(self, issues: list, suggestions: list,
                                 quality_score: float) -> str:
        """生成审查评论"""

        comment_parts = [
            f"## Code Review\n\n**Quality Score: {quality_score:.1f}/100**\n"
        ]

        if issues:
            comment_parts.append(f"\n### Issues Found ({len(issues)}):\n")
            for i, issue in enumerate(issues, 1):
                desc = issue.get('description', str(issue))
                comment_parts.append(f"{i}. {desc}")
        else:
            comment_parts.append("\n### ✓ No major issues found\n")

        if suggestions:
            comment_parts.append(f"\n### Suggestions ({len(suggestions)}):\n")
            for i, suggestion in enumerate(suggestions, 1):
                comment_parts.append(f"{i}. {suggestion}")

        return "\n".join(comment_parts)

    def review(self,
               code: str,
               language: str = "python",
               diff: str = None,
               **kwargs) -> Dict[str, Any]:
        """
        执行代码审查
        
        Args:
            code: 代码内容
            language: 编程语言
            diff: 代码差异
            **kwargs: 其他参数
            
        Returns:
            审查结果字典
        """
        logger.info("开始代码审查...")

        # 创建初始状态
        initial_state = create_initial_state(code=code,
                                             language=language,
                                             diff=diff,
                                             task_type=kwargs.get(
                                                 "task_type", "refinement"),
                                             context=kwargs.get("context", {}))

        # 运行工作流
        final_state = self.graph.invoke(initial_state)

        # 返回结果
        return {
            "quality_score": final_state.get("quality_score"),
            "issues": final_state.get("issues_found", []),
            "suggestions": final_state.get("suggestions", []),
            "review_comment": final_state.get("review_comment"),
            "priority": final_state.get("review_priority"),
            "completed": final_state.get("completed", False),
            "error": final_state.get("error")
        }
