"""
Agent状态定义
定义代码审查Agent的状态结构
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    """代码审查Agent的状态"""

    # 输入
    code: str  # 原始代码
    diff: Optional[str]  # 代码差异
    language: str  # 编程语言
    task_type: str  # 任务类型: refinement/comment/quality

    # 上下文信息
    context: Dict[str, Any]  # 额外上下文（项目、仓库等）

    # Agent处理过程
    messages: Annotated[List[BaseMessage], operator.add]  # 消息历史
    current_stage: str  # 当前阶段

    # 分析结果
    code_features: Optional[Dict[str, Any]]  # 代码特征
    complexity_analysis: Optional[Dict[str, Any]]  # 复杂度分析
    static_analysis: Optional[Dict[str, Any]]  # 静态分析结果

    # 审查结果
    issues_found: List[Dict[str, Any]]  # 发现的问题
    suggestions: List[str]  # 改进建议
    quality_score: Optional[float]  # 质量评分
    review_comment: Optional[str]  # 审查评论

    # 四任务流程字段
    needs_review: Optional[bool]  # 任务一：是否需要评审
    problem_locations: List[Dict[str, Any]]  # 任务二：问题定位
    fixed_code: Optional[str]  # 任务四：修复代码
    task1_output: Optional[Dict[str, Any]]  # 任务一输出
    task2_output: Optional[Dict[str, Any]]  # 任务二输出
    task3_output: Optional[Dict[str, Any]]  # 任务三输出
    task4_output: Optional[Dict[str, Any]]  # 任务四输出

    # 决策信息
    needs_detailed_review: bool  # 是否需要详细审查
    review_priority: str  # 审查优先级: high/medium/low

    # 元信息
    error: Optional[str]  # 错误信息
    completed: bool  # 是否完成


class ReviewStage:
    """审查阶段常量"""
    INIT = "init"  # 初始化
    ANALYZE = "analyze"  # 代码分析
    DETECT = "detect"  # 问题检测
    SUGGEST = "suggest"  # 生成建议
    VALIDATE = "validate"  # 验证建议
    FINALIZE = "finalize"  # 最终整合
    END = "end"  # 结束


class ReviewPriority:
    """审查优先级"""
    HIGH = "high"  # 高优先级（复杂代码、关键修改）
    MEDIUM = "medium"  # 中优先级
    LOW = "low"  # 低优先级（简单修改）


def create_initial_state(
        code: str,
        language: str = "python",
        diff: Optional[str] = None,
        task_type: str = "refinement",
        context: Optional[Dict[str, Any]] = None) -> AgentState:
    """
    创建初始状态
    
    Args:
        code: 代码内容
        language: 编程语言
        diff: 代码差异
        task_type: 任务类型
        context: 上下文信息
        
    Returns:
        初始化的Agent状态
    """
    return AgentState(
        code=code,
        diff=diff,
        language=language,
        task_type=task_type,
        context=context or {},
        messages=[],
        current_stage=ReviewStage.INIT,
        code_features=None,
        complexity_analysis=None,
        static_analysis=None,
        issues_found=[],
        suggestions=[],
        quality_score=None,
        review_comment=None,
        needs_detailed_review=False,
        review_priority=ReviewPriority.MEDIUM,
        # 四任务流程初始值
        needs_review=None,
        problem_locations=[],
        fixed_code=None,
        task1_output=None,
        task2_output=None,
        task3_output=None,
        task4_output=None,
        error=None,
        completed=False)
