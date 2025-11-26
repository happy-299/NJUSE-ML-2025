# Level 3 使用指南

## 快速开始

### 1. 查看数据集统计
```bash
cd NJUSE-ML-2025
python level3/run_agent.py --stats
```

### 2. 运行单个任务

#### 任务一：代码质量评估
```bash
# 使用测试集，限制10个样本
python level3/run_agent.py --dataset task1 --limit 10 --split test

# 使用验证集
python level3/run_agent.py --dataset task1 --limit 50 --split valid
```

**输出**:
- `outputs/level3/task1/predictions_test.json`: 预测结果
- `outputs/level3/task1/metrics_test.json`: 评估指标

#### 任务二：问题代码定位
```bash
python level3/run_agent.py --dataset task2 --limit 10 --split test
```

**输出**:
- `outputs/level3/task2/predictions_test.json`
- `outputs/level3/task2/metrics_test.json`

#### 任务三：评审意见生成
```bash
python level3/run_agent.py --dataset task3 --limit 10 --split test
```

**输出**:
- `outputs/level3/task3/predictions_test.json`
- `outputs/level3/task3/metrics_test.json`

#### 任务四：代码修复
```bash
python level3/run_agent.py --dataset task4 --limit 10 --split test
```

**输出**:
- `outputs/level3/task4/predictions_test.json`
- `outputs/level3/task4/metrics_test.json`

### 3. 运行完整四任务流程

```bash
# 使用 --dataset all 或 --full-pipeline
python level3/run_agent.py --dataset all --limit 10 --split test

# 或
python level3/run_agent.py --full-pipeline --limit 10 --split test
```

**输出**:
- `outputs/level3/task1/metrics_test.json`: 任务一指标
- `outputs/level3/task2/metrics_test.json`: 任务二指标
- `outputs/level3/task3/metrics_test.json`: 任务三指标
- `outputs/level3/task4/metrics_test.json`: 任务四指标
- `outputs/level3/all/detailed_results_test.json`: 所有样本的完整结果
- `outputs/level3/all/summary_results_test.json`: 统计摘要
- `outputs/level3/all/comprehensive_metrics_test.json`: 综合指标

## 配置说明

### 使用本地模型（推荐）

编辑 `level3/.env`:
```bash
LLM_TYPE=local
LOCAL_MODEL_PATH=D:/hehaochuan/codes/machine_learning/NJUSE-ML-2025/level3/codereviewer
DEVICE=cuda  # 或 cpu
```

### 使用 OpenAI API

```bash
LLM_TYPE=openai
OPENAI_API_KEY=your-api-key
```

然后运行:
```bash
python level3/run_agent.py --llm-type openai --model gpt-4o-mini --dataset task1 --limit 10
```

## 输出结构

```
outputs/
└── level3/
    ├── task1/           # 任务一：代码质量评估
    │   ├── predictions_test.json
    │   ├── predictions_valid.json
    │   └── metrics_test.json
    ├── task2/           # 任务二：问题代码定位
    │   ├── predictions_test.json
    │   └── metrics_test.json
    ├── task3/           # 任务三：评审意见生成
    │   ├── predictions_test.json
    │   └── metrics_test.json
    ├── task4/           # 任务四：代码修复
    │   ├── predictions_test.json
    │   └── metrics_test.json
    └── all/             # 完整四任务流程
        ├── detailed_results_test.json
        ├── summary_results_test.json
        └── comprehensive_metrics_test.json
```

## 与 Level 2 的对比

| 维度 | Level 2 (提示工程) | Level 3 (AI Agent) |
|------|-------------------|-------------------|
| 方法 | 静态提示词 | 动态工作流 |
| 决策 | 人工设计 | Agent 自主决策 |
| 任务数 | 单任务为主 | 端到端四任务 |
| 工具使用 | 无 | 代码分析、验证等 |
| 可扩展性 | 低 | 高 |

## 常见问题

**Q: 为什么任务二使用 comment 数据？**
A: 任务二（问题定位）需要评审意见作为输入，所以使用 Comment Generation 数据集。

**Q: 如何只运行特定样本？**
A: 使用 `--limit` 参数限制样本数量。

**Q: 输出文件太大怎么办？**
A: `detailed_results` 包含所有中间输出，如果只需要指标，查看 `metrics_*.json` 即可。

**Q: 如何修改 Agent 决策逻辑？**
A: 编辑 `agent/workflows/full_pipeline_workflow.py` 中的决策函数。
