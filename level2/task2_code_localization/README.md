# Task 2: 问题代码定位 (Level 2 - 提示工程)

## 任务描述

使用大语言模型(LLM)和提示工程完成问题代码定位任务。

### 问题定义

给定:
- 代码变更(old_file + diff_hunk)
- 评审意见(review comment)

输出:
- 需要修改的具体代码行位置(行索引列表)

## 方法与实现

### 数据准备

Task 2 没有独立的数据集,需要从其他任务的数据中提取:
- 使用 Comment Generation 或 Code Refinement 数据集
- 提取包含评审意见的样本
- 标注需要修改的代码行

### 提示词设计

#### System Prompt 设计要点

- 明确角色:代码审查专家,擅长定位问题代码
- 说明任务:根据评审意见定位需要修改的代码行
- 定义输出格式:JSON 格式,包含行索引列表

#### Task Prompt 设计要点

- 提供完整的代码和差异
- 提供评审意见
- 要求 LLM 分析:
  - 评审意见涉及哪些代码部分
  - 哪些行需要修改
  - 输出 0-based 行索引

### 提示词模板

参见 `prompts/` 目录。

## 实验步骤

### 1. 准备数据

```bash
cd level2/task2_code_localization
python prepare_data.py
```

这将从原始数据中提取并标注问题代码行。

### 2. 运行推理

```bash
python inference.py --model gpt-4o-mini --output_file predictions.json
```

### 3. 评估结果

```bash
python evaluate.py --predictions predictions.json
```

## 评估指标

- **Exact Match**: 完全匹配的比例
- **Partial Match**: 部分匹配的比例
- **MRR (Mean Reciprocal Rank)**: 平均倒数排名
- **Top-K Accuracy**: 前 K 个预测中的准确率

## 提示词设计策略

1. **Explicit Instructions**: 明确要求输出行号
2. **Line Numbering**: 在代码中添加行号
3. **Step-by-Step**: 要求 LLM 逐步分析
4. **Examples**: 提供 Few-shot 示例

## 挑战与解决方案

### 挑战 1: 行号对齐

- **问题**: diff 格式的行号不直观
- **解决**: 将 diff 转换为完整代码,添加行号

### 挑战 2: 多处修改

- **问题**: 可能需要修改多个位置
- **解决**: 要求 LLM 输出列表

### 挑战 3: 模糊的评审意见

- **问题**: 有些评审意见不够具体
- **解决**: 引导 LLM 做出合理推断

## 文件说明

- `prompts/system_prompt.txt`: 系统提示词
- `prompts/task_prompt.txt`: 任务提示词模板
- `prepare_data.py`: 数据准备脚本
- `inference.py`: 使用 LLM 进行推理
- `evaluate.py`: 评估脚本
- `README.md`: 本文件

## 数据格式

输入数据格式:
```json
{
    "old_code": "完整的旧代码",
    "diff": "代码差异",
    "comment": "评审意见",
    "language": "编程语言",
    "ground_truth_lines": [2, 5, 7]  // 需要修改的行(0-based)
}
```

输出格式:
```json
{
    "line_indices": [2, 5, 7],
    "confidence": 0.9,
    "reasoning": "详细推理过程"
}
```

## 性能预期

根据 LLM 能力:
- GPT-4: Exact Match ~40%, MRR ~0.6
- GPT-3.5: Exact Match ~30%, MRR ~0.5

## 注意事项

1. 行号从 0 开始计数
2. 只标注新增或修改的行,不标注删除的行
3. 如果评审意见涉及多处,全部标注
4. 对比 Level 3 的 Agent 方法
