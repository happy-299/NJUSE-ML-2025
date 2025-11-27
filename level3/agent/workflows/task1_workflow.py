"""
任务一：代码质量评估（独立工作流）
与 Level 2 完全对齐的实现
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from pathlib import Path
import logging

from ..agents.state import AgentState, create_initial_state
from ..llm_factory import get_llm
from ..config import get_config
from ..prompt_utils import build_quality_estimation_prompt, extract_json_from_response

logger = logging.getLogger(__name__)


class Task1Workflow:
    """任务一：代码质量评估（评审必要性预测）"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.llm = get_llm(self.config)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建简单的单任务图"""
        workflow = StateGraph(AgentState)

        workflow.add_node("quality_assessment", self.quality_assessment)
        workflow.set_entry_point("quality_assessment")
        workflow.add_edge("quality_assessment", END)

        return workflow.compile()

    def quality_assessment(self, state: AgentState) -> AgentState:
        """
        代码质量评估（与 Level 2 完全一致的方法）
        注意：本地 CodeReviewer 模型无法输出标准 JSON，需要使用简化策略
        """
        logger.info("=== 任务一：代码质量评估 ===")

        try:
            old_code = state.get("code", "")
            diff_code = state.get("diff", "")
            language = state.get("language", "code")

            # 使用 Level 2 的 prompt 构建方法
            prompt_dir = Path(__file__).parent.parent / "prompts" / "task1"
            system_prompt_file = str(prompt_dir / "system_prompt.txt")
            task_prompt_file = str(prompt_dir / "task_prompt.txt")

            # 构建标准 prompt
            messages = build_quality_estimation_prompt(
                old_code=old_code,
                diff_code=diff_code,
                language=language,
                system_prompt_file=system_prompt_file,
                task_prompt_file=task_prompt_file,
            )

            # 调用 LLM（将消息格式转换为文本）
            full_prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
            response = self.llm.invoke(full_prompt)

            # 尝试解析标准 JSON 格式
            result = extract_json_from_response(response)

            if result and "needs_review" in result:
                needs_review = result["needs_review"]  # 0 或 1
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
                issues = result.get("issues", [])
                logger.info(f"成功解析 JSON: needs_review={needs_review}")
            else:
                # JSON 解析失败，使用关键词启发式方法（保守策略）
                response_lower = response.lower()

                # 检测"需要评审"的关键词
                needs_keywords = [
                    "needs review", "should review", "requires review",
                    "must review", "bug", "error", "issue", "problem",
                    "security", "vulnerable", "unsafe"
                ]
                no_needs_keywords = [
                    "no review", "not need", "skip review", "safe", "trivial",
                    "formatting only"
                ]

                # 计算关键词得分
                needs_score = sum(1 for kw in needs_keywords
                                  if kw in response_lower)
                no_needs_score = sum(1 for kw in no_needs_keywords
                                     if kw in response_lower)

                if no_needs_score > needs_score:
                    needs_review = 0
                    confidence = min(0.6 + no_needs_score * 0.1, 0.9)
                    reasoning = f"Heuristic: No review needed (score: {no_needs_score} vs {needs_score})"
                else:
                    needs_review = 1  # 保守策略：默认需要评审
                    confidence = min(0.5 + needs_score * 0.1, 0.9)
                    reasoning = f"Heuristic: Review needed (score: {needs_score} vs {no_needs_score})"

                issues = []
                logger.warning(
                    f"Failed to parse JSON, using heuristic: needs_review={needs_review}, confidence={confidence:.2f}"
                )
                logger.debug(f"Response preview: {response[:300]}")

            # 保存到 state
            state["needs_review"] = bool(needs_review)
            state["task1_output"] = {
                "needs_review": needs_review,  # 标准 0/1 格式
                "confidence": confidence,
                "reasoning": reasoning,
                "issues": issues,
                "raw_response": response,
                "completed": True,
            }

            logger.info(
                f"评审必要性: {'需要' if needs_review else '不需要'} (confidence: {confidence:.2f})"
            )

        except Exception as e:
            logger.error(f"任务一失败: {e}", exc_info=True)
            state["needs_review"] = True
            state["task1_output"] = {
                "needs_review": 1,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "issues": [],
                "raw_response": "",
                "completed": False,
            }
            state["error"] = str(e)

        return state

    def run(self,
            code: str,
            diff: str = "",
            language: str = "code") -> Dict[str, Any]:
        """
        运行质量评估

        Args:
            code: 代码内容
            diff: 代码差异（可选）
            language: 编程语言

        Returns:
            评估结果
        """
        initial_state = create_initial_state(code=code,
                                             diff=diff,
                                             language=language)
        final_state = self.graph.invoke(initial_state)

        return {
            "needs_review": final_state.get("needs_review"),
            "task1_output": final_state.get("task1_output"),
            "error": final_state.get("error"),
        }
