"""
Agent 工作流状态定义
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    """Agent 工作流的状态"""
    
    # 输入数据
    sample_id: int
    old_code: str
    new_code: str
    diff: str
    language: str
    
    # 任务一：代码质量评估
    quality_score: float
    needs_review: bool
    quality_reasoning: str
    quality_issues: List[str]
    
    # 任务二：问题代码定位
    problem_locations: List[Dict[str, Any]]
    problem_count: int
    localization_confidence: float
    localization_reasoning: str
    
    # 任务三：评审意见生成
    review_comment: str
    comment_length: int
    
    # 任务四：代码修复
    fixed_code: str
    fix_applied: bool
    fix_verification: str
    
    # Agent 决策记录
    agent_decisions: Dict[str, str]
    
    # 元数据
    ground_truth: Dict[str, Any]
    error: Optional[str]


class TaskResult(TypedDict, total=False):
    """单个任务的结果"""
    
    success: bool
    output: Any
    confidence: float
    reasoning: str
    error: Optional[str]
