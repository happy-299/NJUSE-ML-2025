"""
完整的四任务端到端代码审查工作流
任务一 → 任务二 → 任务三 → 任务四
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from pathlib import Path
import logging

from ..agents.state import AgentState, ReviewStage, create_initial_state
from ..llm_factory import get_llm
from ..config import get_config
from ..prompt_utils import build_quality_estimation_prompt, extract_json_from_response

logger = logging.getLogger(__name__)


class FullPipelineWorkflow:
    """完整四任务端到端工作流"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.llm = get_llm(self.config)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建四任务串联的状态图"""
        workflow = StateGraph(AgentState)

        # 添加四个任务节点
        workflow.add_node("task1_quality_assessment",
                          self.task1_quality_assessment)
        workflow.add_node("task2_problem_localization",
                          self.task2_problem_localization)
        workflow.add_node("task3_review_generation",
                          self.task3_review_generation)
        workflow.add_node("task4_code_fixing", self.task4_code_fixing)
        workflow.add_node("finalize", self.finalize_pipeline)

        # 设置入口
        workflow.set_entry_point("task1_quality_assessment")

        # 任务一 → 条件分支
        workflow.add_conditional_edges(
            "task1_quality_assessment", self._should_continue_review, {
                "continue": "task2_problem_localization",
                "skip": "finalize"
            })

        # 任务二 → 条件分支（根据问题严重程度决策）
        workflow.add_conditional_edges(
            "task2_problem_localization", self._should_generate_fix, {
                "fix_needed": "task3_review_generation",
                "review_only": "task3_review_generation",
                "no_issues": "finalize"
            })

        # 任务三 → 条件分支（根据评审结果决定是否修复）
        workflow.add_conditional_edges("task3_review_generation",
                                       self._should_fix_code, {
                                           "fix": "task4_code_fixing",
                                           "skip_fix": "finalize"
                                       })

        # 任务四 → 结束
        workflow.add_edge("task4_code_fixing", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def _should_generate_fix(self, state: AgentState) -> str:
        """
        Agent决策：根据问题数量和严重程度决定下一步
        """
        problem_locations = state.get("problem_locations", [])
        problem_count = len(problem_locations)

        # 调试：打印state中的关键信息
        logger.debug(
            f"[DEBUG] _should_generate_fix - problem_locations类型: {type(problem_locations)}"
        )
        logger.debug(
            f"[DEBUG] _should_generate_fix - problem_locations内容: {problem_locations[:2] if problem_locations else 'empty'}"
        )
        logger.debug(
            f"[DEBUG] _should_generate_fix - problem_count: {problem_count}")

        if problem_count == 0:
            logger.info("🤖 Agent决策: 未发现问题，但仍生成基础评审")
            # 修改：即使没问题也生成评审（改善用户体验）
            return "review_only"
        elif problem_count <= 2:
            logger.info(f"🤖 Agent决策: 发现{problem_count}个问题，生成评审意见但不自动修复")
            return "review_only"
        else:
            logger.info(f"🤖 Agent决策: 发现{problem_count}个问题，需要生成评审和修复建议")
            return "fix_needed"

    def _should_fix_code(self, state: AgentState) -> str:
        """
        Agent决策：根据评审意见判断是否需要生成修复代码
        """
        problem_count = len(state.get("problem_locations", []))
        review_comment = state.get("review_comment", "")

        # 检查评审中是否包含严重问题的关键词
        critical_keywords = [
            'bug', 'error', 'security', 'critical', 'must fix', 'vulnerable'
        ]
        has_critical = any(keyword in review_comment.lower()
                           for keyword in critical_keywords)

        if has_critical or problem_count >= 3:
            logger.info("🤖 Agent决策: 发现严重问题，生成修复代码")
            return "fix"
        else:
            logger.info("🤖 Agent决策: 问题不严重，仅提供评审建议，不生成修复代码")
            return "skip_fix"

    def _should_continue_review(self, state: AgentState) -> str:
        """
        Agent决策：判断是否需要继续深度评审
        这是Agent的核心 - 自主决策下一步行动
        """
        quality_score = state.get("quality_score", 50)
        needs_review = state.get("needs_review", True)

        # Agent决策逻辑（调整阈值以适应测试）：
        # 1. 极高质量代码(>95) + 不需要评审 -> 跳过后续任务
        # 2. 中等质量(60-95) -> 执行问题定位，但可能跳过修复
        # 3. 低质量(<60) -> 执行完整流程

        if quality_score >= 95 and not needs_review:
            logger.info(f"🤖 Agent决策: 代码质量极优({quality_score})，无需深度评审，直接通过")
            return "skip"
        elif quality_score >= 60:
            logger.info(f"🤖 Agent决策: 代码质量中等({quality_score})，执行标准评审流程")
            return "continue"
        else:
            logger.info(f"🤖 Agent决策: 代码质量较低({quality_score})，需要全面评审和修复")
            return "continue"

    def task1_quality_assessment(self, state: AgentState) -> AgentState:
        """
        任务一：代码质量评估（评审必要性预测）
        使用 Level 2 的标准方法和 prompt
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

            # 解析响应（Level 2 格式）
            result = extract_json_from_response(response)

            if result and "needs_review" in result:
                needs_review = result["needs_review"]  # 0 或 1
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "")
                issues = result.get("issues", [])
            else:
                # 解析失败，使用保守策略
                needs_review = 1  # 默认需要评审
                confidence = 0.5
                reasoning = "Failed to parse LLM response"
                issues = []
                logger.warning(f"Failed to parse response: {response[:200]}")

            # 为了兼容后续流程，保留 quality_score（但不再用于指标计算）
            quality_score = 50 if needs_review == 1 else 85

            state["needs_review"] = bool(needs_review)
            state["quality_score"] = quality_score
            state["task1_output"] = {
                "needs_review": needs_review,  # 标准 0/1 格式
                "confidence": confidence,
                "reasoning": reasoning,
                "issues": issues,
                "raw_response": response,
            }

            logger.info(
                f"评审必要性: {'需要' if needs_review else '不需要'} (confidence: {confidence:.2f})"
            )

        except Exception as e:
            logger.error(f"任务一失败: {e}")
            state["needs_review"] = True
            state["quality_score"] = 50
            state["task1_output"] = {
                "needs_review": 1,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "issues": [],
                "raw_response": "",
            }
            state["error"] = str(e)

        return state

    def _analyze_code_complexity(self, code: str, diff: str) -> dict:
        """
        Agent工具：分析代码复杂度
        """
        text = diff if diff else code
        lines = len([l for l in text.split('\n') if l.strip()])

        # 简单的复杂度评估
        complexity_indicators = {
            'loops': text.count('for ') + text.count('while '),
            'conditions': text.count('if ') + text.count('elif '),
            'functions': text.count('def ') + text.count('function '),
            'nested_depth': text.count('    ') // 4  # 简化的嵌套深度
        }

        total_complexity = sum(complexity_indicators.values())

        if total_complexity > 20 or lines > 200:
            complexity = 'high'
        elif total_complexity > 10 or lines > 100:
            complexity = 'medium'
        else:
            complexity = 'low'

        return {
            'complexity': complexity,
            'lines': lines,
            **complexity_indicators
        }

    def task2_problem_localization(self, state: AgentState) -> AgentState:
        """
        任务二：问题代码行定位
        输出：问题代码行列表 [{line_no, code_snippet, issue_type}]
        """
        logger.info("=== 任务二：问题代码定位 ===")

        try:
            code = state.get("code", "")
            diff = state.get("diff", "")

            prompt = f"""You are an expert code analyzer. Carefully examine the following code and identify ALL problematic lines.

## Code to Analyze
```
{code[:2500] if code else diff[:2500]}
```

## Analysis Guidelines
Identify issues in these categories:

### 1. BUGS (Critical)
- Null/undefined reference errors
- Off-by-one errors in loops/arrays
- Race conditions or concurrency issues
- Resource leaks (unclosed files, connections)
- Type mismatches or casting errors
- Logic errors (wrong operators, incorrect conditions)

### 2. SECURITY (Critical)
- SQL injection vulnerabilities
- Cross-site scripting (XSS) risks
- Hardcoded credentials or secrets
- Improper input validation
- Insecure cryptography usage
- Authentication/authorization bypasses

### 3. PERFORMANCE (Important)
- Inefficient algorithms (O(n²) where O(n) possible)
- Unnecessary database queries in loops
- Memory leaks
- Blocking I/O operations
- Redundant computations

### 4. CODE QUALITY (Important)
- Code duplication
- Magic numbers without constants
- Deep nesting (>3 levels)
- Long methods (>50 lines)
- Poor variable naming
- Missing error handling
- Unused variables or imports

## Output Format
For EACH issue found, output EXACTLY in this format:
LINE <number>: [<CATEGORY>] <specific_issue_description>

Example:
LINE 15: [BUG] Potential null pointer exception when user is undefined
LINE 23: [SECURITY] SQL query vulnerable to injection attack
LINE 45: [PERFORMANCE] O(n²) loop could be optimized with Set lookup
LINE 67: [QUALITY] Magic number 86400 should be named constant SECONDS_PER_DAY

## Important
- Be specific: mention variable names, line numbers, exact issues
- Prioritize critical bugs and security issues
- If no issues found, respond with: "No significant issues detected."
- Do NOT provide fixes yet, only identify problems

Your analysis:"""

            response = self.llm.invoke(prompt)

            # 解析定位结果 - 更灵活的解析
            problem_locations = []
            lines = response.strip().split('\n')

            # 尝试多种格式的解析
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 格式1: LINE X: ...
                if 'LINE' in line.upper():
                    problem_locations.append({
                        "location": line,
                        "description": line
                    })
                # 格式2: 包含问题类型关键词
                elif any(keyword in line.lower() for keyword in [
                        'bug', 'error', 'issue', 'problem', 'fix', 'warning',
                        'security', 'performance'
                ]):
                    problem_locations.append({
                        "location": "General issue",
                        "description": line
                    })

            # 如果模型输出了内容但没有匹配到格式，将整个响应作为一个问题
            if not problem_locations and response.strip() and len(
                    response.strip()) > 10:
                problem_locations.append({
                    "location": "Code analysis",
                    "description": response.strip()[:200]
                })

            state["problem_locations"] = problem_locations
            state["task2_output"] = {
                "problem_count": len(problem_locations),
                "locations": problem_locations
            }

            logger.info(f"定位到 {len(problem_locations)} 个问题位置")

        except Exception as e:
            logger.error(f"任务二失败: {e}")
            state["problem_locations"] = []
            state["error"] = str(e)

        return state

    def task3_review_generation(self, state: AgentState) -> AgentState:
        """
        任务三：评审意见生成
        输入：任务二的定位结果
        输出：自然语言评审意见
        """
        logger.info("=== 任务三：评审意见生成 ===")

        try:
            problem_locations = state.get("problem_locations", [])
            code = state.get("code", "")
            diff = state.get("diff", "")

            if not problem_locations:
                # 即使没有识别到具体问题，也生成基于diff的评审
                prompt = f"""You are a senior software engineer conducting a code review. 

## Code Change
```
{diff[:2000] if diff else code[:2000]}
```

## Review Focus Areas
Since no critical issues were detected, provide constructive feedback on:

1. **Code Quality**
   - Is the code readable and maintainable?
   - Are naming conventions clear and consistent?
   - Is the code properly structured?

2. **Best Practices**
   - Does it follow language/framework conventions?
   - Are there better patterns or idioms to use?
   - Is error handling appropriate?

3. **Potential Improvements**
   - Could complexity be reduced?
   - Are there opportunities for abstraction?
   - Would additional comments help clarity?

4. **Testing Considerations**
   - Is this code easily testable?
   - Are edge cases handled?
   - What tests should be added?

## Response Format
Provide a concise, professional review (2-4 sentences) that:
- Acknowledges what's good about the code
- Suggests 1-2 meaningful improvements (if any)
- Uses encouraging, constructive tone

Example: "The code is well-structured with clear variable names. Consider extracting the validation logic into a separate function for better testability. Overall, this is a solid implementation."

Your review:"""
                review_comment = self.llm.invoke(prompt)
            else:
                # 基于定位结果生成评审意见
                issues_text = chr(10).join([
                    f"{i+1}. {loc.get('description', '')}"
                    for i, loc in enumerate(problem_locations[:8])
                ])
                prompt = f"""You are a senior software engineer providing actionable code review feedback.

## Identified Issues ({len(problem_locations)} total)
{issues_text}

## Code Context
```
{diff[:1500] if diff else code[:1500]}
```

## Task
Generate a professional, constructive code review comment that addresses the identified issues.

## Requirements

### Structure
1. **Opening** (1 sentence): Brief assessment of overall code quality
2. **Issues** (2-4 points): For each major issue:
   - Clearly explain the problem
   - Explain why it's problematic (impact/risk)
   - Provide a specific solution or best practice
3. **Positive Note** (optional): Acknowledge any good aspects
4. **Closing** (1 sentence): Encouraging summary

### Tone Guidelines
- Be respectful and constructive (avoid "you should" → use "consider" or "suggest")
- Focus on code, not the developer
- Explain the "why" behind suggestions
- Use specific examples when possible
- Balance criticism with recognition

### Example Format
```
Thanks for the submission! I've identified several areas for improvement:

1. **Security Concern (Line 15)**: The SQL query is vulnerable to injection attacks. Consider using parameterized queries: `cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))`

2. **Performance Issue (Line 23)**: The nested loop creates O(n²) complexity. Using a Set for lookup would improve this to O(n): `valid_ids = set(all_ids)`

3. **Code Quality (Line 45)**: The magic number 86400 reduces readability. Define it as a constant: `SECONDS_PER_DAY = 86400`

The overall structure is clean and the error handling is well-implemented. Once these issues are addressed, this will be ready to merge!
```

## Your Review Comment
(Write 3-6 sentences covering the major issues):"""

                review_comment = self.llm.invoke(prompt)

            state["review_comment"] = review_comment
            state["task3_output"] = {
                "review_comment": review_comment,
                "based_on_locations": len(problem_locations)
            }

            logger.info(f"生成评审意见: {review_comment[:100]}...")

        except Exception as e:
            logger.error(f"任务三失败: {e}")
            state["review_comment"] = "Error generating review comment."
            state["error"] = str(e)

        return state

    def _verify_fixed_code(self, old_code: str, fixed_code: str,
                           problems: list) -> dict:
        """Agent工具：验证修复后的代码质量"""
        # 简单启发式验证
        verification = {
            "is_valid": True,
            "issues": [],
            "quality_improved": False
        }

        # 检查1: 代码是否过短（可能是不完整的修复）
        if len(fixed_code.strip()) < len(old_code.strip()) * 0.5:
            verification["is_valid"] = False
            verification["issues"].append(
                "Fixed code too short - incomplete fix")

        # 检查2: 是否包含明显的错误标记
        error_markers = ["TODO", "FIXME", "ERROR", "BUG", "XXX"]
        if any(marker in fixed_code.upper() for marker in error_markers):
            verification["issues"].append("Contains error markers")

        # 检查3: 基于问题数量判断改进程度
        if len(problems) >= 3:
            # 重大问题应该有显著变化
            if abs(len(fixed_code) - len(old_code)) < 20:
                verification["issues"].append(
                    "Minimal changes for critical issues")
            else:
                verification["quality_improved"] = True
        else:
            verification["quality_improved"] = True

        return verification

    def task4_code_fixing(self, state: AgentState) -> AgentState:
        """
        任务四：修复代码生成（增强Agent能力：验证+重试）
        输入：任务二的定位 + 任务三的评审意见
        输出：修复后的代码
        Agent特性：
        - 验证修复质量
        - 失败时自动重试（最多2次）
        - 记录决策过程
        """
        logger.info("=== 任务四：修复代码生成（Agent验证模式） ===")

        try:
            review_comment = state.get("review_comment", "")
            problem_locations = state.get("problem_locations", [])
            old_code = state.get("old_code", "") or state.get("code", "")

            if not problem_locations:
                fixed_code = old_code
                state["task4_output"] = {
                    "fixed_code": fixed_code,
                    "changes_made": False,
                    "verified": True,
                    "retry_count": 0
                }
            else:
                # Agent重试机制
                max_retries = 2
                retry_count = 0
                verified = False
                fixed_code = None
                verification_result = None

                while retry_count <= max_retries and not verified:
                    logger.info(
                        f"🤖 Agent修复尝试 {retry_count + 1}/{max_retries + 1}")

                    # 第一次尝试：标准修复
                    if retry_count == 0:
                        problems_summary = chr(10).join([
                            f"- {loc.get('description', '')[:100]}"
                            for loc in problem_locations[:5]
                        ])
                        prompt = f"""You are an expert programmer tasked with fixing code issues.

## Original Code
```
{old_code[:2000]}
```

## Identified Problems
{problems_summary}

## Review Feedback
{review_comment[:500]}

## Task
Generate the COMPLETE, CORRECTED version of the code that addresses all identified issues.

## Requirements
1. **Fix All Issues**: Address every problem mentioned above
2. **Maintain Functionality**: Preserve the original code's intended behavior
3. **Complete Code**: Return the FULL corrected code, not just snippets
4. **Best Practices**: Apply proper coding standards and patterns
5. **No Placeholders**: No TODOs, FIXMEs, or incomplete sections

## Quality Checklist
- [ ] All bugs fixed
- [ ] Security vulnerabilities patched
- [ ] Performance improvements applied
- [ ] Code quality enhanced (naming, structure, comments)
- [ ] No syntax errors
- [ ] All functions/logic complete

## Output Format
Respond with ONLY the corrected code (no explanations or markdown):
"""
                    else:
                        # 重试时提供更详细的指导
                        previous_issues = ", ".join(
                            verification_result["issues"])
                        problems_summary = chr(10).join([
                            f"- {loc.get('description', '')[:100]}"
                            for loc in problem_locations[:5]
                        ])
                        prompt = f"""⚠️ RETRY ATTEMPT {retry_count + 1}/{max_retries + 1}

Your previous fix failed verification: {previous_issues}

## Original Code (MUST preserve all functionality)
```
{old_code[:2000]}
```

## CRITICAL: Problems That MUST Be Fixed
{problems_summary}

## Review Guidance
{review_comment[:500]}

## Why Previous Attempt Failed
{previous_issues}

## STRICT Requirements
1. Fix ALL {len(problem_locations)} identified issues
2. Return COMPLETE code (not shortened/truncated)
3. Ensure code length is appropriate (at least {int(len(old_code)*0.8)} characters)
4. No TODO, FIXME, or placeholder comments
5. Syntactically correct and runnable
6. Preserve all original functions and logic

## Common Mistakes to Avoid
- Returning only the changed parts (need FULL code)
- Incomplete implementations
- Adding placeholder comments
- Making code shorter by removing functionality

## Output
Generate the COMPLETE, FULLY CORRECTED code now:
"""

                    fixed_code = self.llm.invoke(prompt)

                    # 🔧 Agent工具：验证修复质量
                    verification_result = self._verify_fixed_code(
                        old_code, fixed_code, problem_locations)
                    logger.info(
                        f"[Agent工具] 代码验证 - Valid={verification_result['is_valid']}, Improved={verification_result['quality_improved']}"
                    )

                    if verification_result["is_valid"] and (
                            verification_result["quality_improved"]
                            or len(problem_locations) <= 2):
                        verified = True
                        logger.info(f"✓ Agent决策: 修复通过验证")
                    else:
                        logger.warning(
                            f"✗ Agent决策: 修复未通过验证 - {verification_result['issues']}"
                        )
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.warning(f"🤖 Agent决策: 达到最大重试次数，使用最后一次结果")

                state["task4_output"] = {
                    "fixed_code":
                    fixed_code,
                    "changes_made":
                    True,
                    "original_code":
                    old_code,
                    "verified":
                    verified,
                    "retry_count":
                    retry_count,
                    "verification_issues":
                    verification_result["issues"]
                    if verification_result else []
                }

            state["fixed_code"] = fixed_code

            logger.info(f"生成修复代码: {len(fixed_code)} 字符")

        except Exception as e:
            logger.error(f"任务四失败: {e}")
            state["fixed_code"] = state.get("code", "")
            state["error"] = str(e)

        return state

    def finalize_pipeline(self, state: AgentState) -> AgentState:
        """整合所有任务结果"""
        logger.info("=== 整合四任务结果 ===")

        # 调试：打印关键字段（安全处理None值）
        logger.debug(
            f"[DEBUG] finalize - problem_locations: {len(state.get('problem_locations') or [])}"
        )
        logger.debug(
            f"[DEBUG] finalize - review_comment: {len(state.get('review_comment') or '')}"
        )
        logger.debug(
            f"[DEBUG] finalize - fixed_code: {len(state.get('fixed_code') or '')}"
        )

        # 确保所有必需字段都有默认值
        if "review_comment" not in state or state["review_comment"] is None:
            state["review_comment"] = ""
        if "fixed_code" not in state or state["fixed_code"] is None:
            state["fixed_code"] = state.get("code", "")
        if "problem_locations" not in state or state[
                "problem_locations"] is None:
            state["problem_locations"] = []
        if "task3_output" not in state:
            state["task3_output"] = {
                "review_comment": "",
                "based_on_locations": 0
            }
        if "task4_output" not in state:
            state["task4_output"] = {
                "fixed_code": state.get("code", ""),
                "changes_made": False
            }

        state["completed"] = True
        state["pipeline_result"] = {
            "task1": state.get("task1_output", {}),
            "task2": state.get("task2_output", {}),
            "task3": state.get("task3_output", {}),
            "task4": state.get("task4_output", {})
        }

        logger.info("四任务流程完成")
        return state

    def review(self,
               code: str = "",
               language: str = "python",
               diff: str = "",
               old_code: str = "",
               new_code: str = "",
               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行完整的四任务评审流程
        
        Args:
            code: 代码内容
            language: 编程语言
            diff: 代码差异
            old_code: 旧代码
            new_code: 新代码
            context: 额外上下文（包含参考答案）
            
        Returns:
            包含所有任务结果的字典
        """
        # 创建初始状态
        initial_state = create_initial_state(code=code,
                                             language=language,
                                             diff=diff)

        # 添加额外字段
        initial_state["old_code"] = old_code
        initial_state["new_code"] = new_code
        initial_state["context"] = context or {}

        # 运行工作流
        final_state = self.graph.invoke(initial_state)

        # 返回结果（包含Agent决策信息）
        return {
            "quality_score": final_state.get("quality_score", 0),
            "needs_review": final_state.get("needs_review", True),
            "problem_locations": final_state.get("problem_locations", []),
            "review_comment": final_state.get("review_comment", ""),
            "fixed_code": final_state.get("fixed_code", ""),
            "completed": final_state.get("completed", False),
            "pipeline_result": final_state.get("pipeline_result", {}),
            # Agent决策信息
            "task1_output": final_state.get("task1_output", {}),
            "task2_output": final_state.get("task2_output", {}),
            "task3_output": final_state.get("task3_output", {}),
            "task4_output": final_state.get("task4_output", {}),
            "error": final_state.get("error")
        }
