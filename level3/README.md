# Level 3: AI Agent 代码评审系统

基于 LangGraph 构建的 AI Agent 系统，实现端到端的代码评审自动化流程。

## 系统架构

本系统使用 AI Agent 架构，包含以下核心组件：

### 1. Agent 核心 (`agent/`)

- **状态管理** (`agents/state.py`): 定义 Agent 工作流状态
- **工作流** (`workflows/`): 实现四任务端到端流程
  - `full_pipeline_workflow.py`: 完整四任务流程（质量评估→问题定位→评审生成→代码修复）
  - `review_workflow.py`: 简化评审流程
- **工具集** (`tools/`): Agent 使用的工具函数
  - `code_analyzer.py`: 代码分析工具（复杂度、行数等）
- **LLM 工厂** (`llm_factory.py`): 支持多种 LLM 后端
  - 本地 CodeReviewer 模型
  - OpenAI API
  - Ollama
- **数据加载** (`data_loader.py`): 加载和处理数据集
- **评价指标** (`metrics.py`): 计算四任务的评价指标

### 2. 提示词 (`prompts/`)

- `simple_prompts.py`: 四任务的提示词模板

### 3. 模型文件 (`codereviewer/`)

本地 CodeReviewer 模型权重（基于 T5 架构）

## 四任务流程

### 任务一：代码质量评估
- **输入**: 代码 + diff
- **输出**: 质量评分 (0-100) + 是否需要评审
- **工具**: 代码复杂度分析
- **决策**: 根据评分决定是否进入任务二

### 任务二：问题代码定位
- **输入**: 代码 + diff
- **输出**: 问题位置列表 (文件名、行号、描述)
- **决策**: 根据问题数量决定是否进入任务三

### 任务三：评审意见生成
- **输入**: 代码 + diff + 问题位置
- **输出**: 评审意见文本
- **决策**: 根据问题严重程度决定是否进入任务四

### 任务四：代码修复
- **输入**: 代码 + diff + 问题位置 + 评审意见
- **输出**: 修复后的代码
- **验证**: 可选的代码验证和质量检查

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r ../requirements.txt

# 或安装 level3 特定依赖
pip install langchain langchain-openai langgraph transformers torch
```

### 2. 配置 LLM

复制 `.env.example` 为 `.env` 并修改：

```bash
# 使用本地模型（推荐）
LLM_TYPE=local
LOCAL_MODEL_PATH=D:/hehaochuan/codes/machine_learning/NJUSE-ML-2025/level3/codereviewer
DEVICE=cuda

# 或使用 OpenAI API
LLM_TYPE=openai
OPENAI_API_KEY=your-api-key
```

### 3. 运行 Agent

#### 完整四任务流程

```bash
# 使用本地模型
python run_agent.py --llm-type local --dataset quality --limit 10 --full-pipeline

# 使用 OpenAI
python run_agent.py --llm-type openai --model gpt-4o-mini --dataset quality --limit 5
```

#### 简单评审流程（仅任务三）

```bash
python run_agent.py --dataset comment --limit 10
```

#### 查看数据集统计

```bash
python run_agent.py --stats
```

### 4. 参数说明

- `--llm-type`: LLM 类型 (local/openai/ollama)
- `--model`: 模型名称 (gpt-4o-mini/gpt-4/etc.)
- `--dataset`: 数据集类型 (quality/comment/refinement)
- `--limit`: 限制样本数量
- `--full-pipeline`: 使用完整四任务流程
- `--stats`: 显示数据集统计信息

## 输出文件

结果保存在 `agent/output/` 目录：

### 1. 详细结果 (`detailed_results_*.json`)

包含每个样本的完整模型输出：

```json
{
  "sample_id": 1,
  "input": {...},
  "task1_quality_assessment": {
    "quality_score": 85,
    "needs_review": false,
    "model_output": 85
  },
  "task2_problem_localization": {
    "problem_locations": [...],
    "problem_count": 1,
    "model_output": [...]
  },
  "task3_review_generation": {
    "review_comment": "...",
    "model_output": "..."
  },
  "task4_code_fixing": {
    "fixed_code": "...",
    "model_output": "..."
  },
  "agent_decisions": {
    "task1_to_task2": "continue",
    "task2_to_task3": "continue",
    "task4_verification": "passed"
  }
}
```

### 2. 统计摘要 (`summary_results_*.json`)

快速查看的统计信息。

### 3. 综合指标 (`comprehensive_metrics_*.json`)

四个任务的评价指标：

```json
{
  "task1_quality_assessment": {
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.90,
    "f1_macro": 0.86,
    "sample_count": 100
  },
  "task2_problem_localization": {
    "accuracy": 0.65,
    "precision": 0.60,
    "recall": 0.70,
    "mrr": 0.72
  },
  "task3_review_generation": {
    "total_samples": 100,
    "generated_reviews": 98,
    "generation_rate": 98.0,
    "avg_review_length": 156.3
  },
  "task4_code_fixing": {
    "total_samples": 100,
    "generated_fixes": 85,
    "generation_rate": 85.0,
    "avg_fixed_length": 1234.5
  }
}
```

## Agent 工作流可视化

```
┌─────────────────────────────────────────────────────────┐
│                  开始 (Start)                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  任务一：代码质量评估                                    │
│  - 使用 code_analyzer 工具分析复杂度                    │
│  - LLM 评估代码质量 (0-100)                             │
│  - 决策：质量 < 阈值 → 继续                             │
└────────────────────┬────────────────────────────────────┘
                     │ needs_review = True
                     ▼
┌─────────────────────────────────────────────────────────┐
│  任务二：问题代码定位                                    │
│  - LLM 定位代码中的问题                                 │
│  - 提取问题位置（文件、行号、描述）                      │
│  - 决策：问题数 > 0 → 继续                              │
└────────────────────┬────────────────────────────────────┘
                     │ problem_count > 0
                     ▼
┌─────────────────────────────────────────────────────────┐
│  任务三：评审意见生成                                    │
│  - 基于问题位置生成评审意见                             │
│  - LLM 生成自然语言评论                                 │
│  - 决策：严重问题 → 任务四，否则 → 结束                 │
└────────────────────┬────────────────────────────────────┘
                     │ severe_issues
                     ▼
┌─────────────────────────────────────────────────────────┐
│  任务四：代码修复                                        │
│  - LLM 生成修复后的代码                                 │
│  - （可选）验证修复代码质量                             │
│  - 决策：验证失败 → 重试（最多3次）                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   结束 (End)                            │
│  - 整合四任务结果                                        │
│  - 保存完整输出                                          │
└─────────────────────────────────────────────────────────┘
```

## 性能优化

1. **批处理**: 使用 `--limit` 参数限制样本数，避免长时间运行
2. **缓存**: 已处理的结果会保存，可复用
3. **并行**: 多样本处理可并行（需要修改代码）
4. **模型优化**: 
   - 使用 float16 加速推理
   - 调整 `max_length` 和 `num_beams` 参数

## 常见问题

### Q1: CUDA out of memory?

调整 batch size 或使用 CPU：
```bash
DEVICE=cpu python run_agent.py ...
```

### Q2: 模型加载失败?

检查 `.env` 中的 `LOCAL_MODEL_PATH` 是否正确。

### Q3: 如何使用自己的 LLM?

在 `llm_factory.py` 中添加自定义 LLM 类：

```python
class CustomLLM(BaseLLM):
    def _call(self, prompt: str, **kwargs) -> str:
        # 实现你的 LLM 调用逻辑
        pass
```

### Q4: 如何调整 Agent 决策逻辑?

修改 `workflows/full_pipeline_workflow.py` 中的决策函数：
- `_should_continue_to_task2()`
- `_should_continue_to_task3()`
- `_should_generate_fix()`

## 扩展开发

### 添加新工具

在 `tools/` 目录添加新工具：

```python
# tools/custom_tool.py
def my_custom_tool(code: str) -> dict:
    """自定义工具"""
    # 实现工具逻辑
    return {"result": "..."}
```

在工作流中使用：

```python
from ..tools.custom_tool import my_custom_tool

# 在 workflow 中调用
result = my_custom_tool(state["code"])
```

### 修改提示词

编辑 `prompts/simple_prompts.py`：

```python
QUALITY_ASSESSMENT_PROMPT = """
你的新提示词...
"""
```

## 贡献指南

1. 在独立分支开发新功能
2. 添加单元测试
3. 更新文档
4. 提交 PR

## 参考资料

- [LangGraph 文档](https://python.langchain.com/docs/langgraph)
- [LangChain 文档](https://python.langchain.com/)
- [CodeReviewer 论文](https://arxiv.org/abs/2203.09095)

## 作者

NJUSE ML 2025 - Level 3 团队
