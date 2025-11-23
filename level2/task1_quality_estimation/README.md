# Task 1: 代码质量评估 (Level 2 - 提示工程)

## 任务描述

使用大语言模型(LLM)和提示工程(Prompt Engineering)完成代码质量评估任务。

### 问题定义

给定代码变更(old_file + diff_hunk),通过精心设计的提示词,让 LLM 判断该变更是否需要人工评审。

## 方法与实现

### LLM 选择

- **推荐模型**: GPT-4o-mini, GPT-4, Claude-3, 或开源模型
- **API 配置**: 在 `config.py` 中配置 API 密钥

### 提示词设计

提示词分为两部分:

1. **System Prompt** (`prompts/system_prompt.txt`): 定义 LLM 的角色和任务
2. **Task Prompt** (`prompts/task_prompt.txt`): 提供具体的代码和问题

#### System Prompt 设计要点

- 明确 LLM 的角色:代码审查专家
- 说明任务目标:判断代码变更质量
- 定义输出格式:JSON 格式,包含判断结果和理由

#### Task Prompt 设计要点

- 提供完整的旧代码和代码差异
- 要求 LLM 从多个维度分析:
  - 代码正确性
  - 代码规范性
  - 潜在问题
  - 是否需要人工评审
- 使用 Few-shot Learning 提供示例

### 提示词模板

参见 `prompts/` 目录下的文件。

## 实验步骤

### 1. 配置 API 密钥

```bash
# 设置环境变量
export OPENAI_API_KEY="your_api_key_here"
```

或在 `config.py` 中直接配置。

### 2. 运行推理

```bash
cd level2/task1_quality_estimation
python inference.py --model gpt-4o-mini --output_file predictions.json
```

### 3. 评估结果

```bash
python evaluate.py --predictions predictions.json --labels ../../data/raw/Diff_Quality_Estimation/cls-test.jsonl
```

## 评估指标

- Accuracy (准确率)
- Precision (精确度)
- Recall (召回率)
- F1-Score (Macro 平均)

## 提示词优化策略

1. **Few-shot Learning**: 在提示词中添加 2-3 个示例
2. **Chain-of-Thought**: 要求 LLM 逐步分析
3. **Role-playing**: 让 LLM 扮演经验丰富的代码审查者
4. **Output Formatting**: 明确指定输出格式(JSON)
5. **多轮对话**: 对不确定的样本进行多轮询问

## 提示词迭代记录

记录每次提示词修改和对应的性能变化,用于实验报告。

| 版本 | 主要修改      | Accuracy | F1-Score |
| ---- | ------------- | -------- | -------- |
| v1   | 基础提示词    | -        | -        |
| v2   | 添加 Few-shot | -        | -        |
| v3   | 添加 CoT      | -        | -        |

## 文件说明

- `prompts/system_prompt.txt`: 系统提示词
- `prompts/task_prompt.txt`: 任务提示词模板
- `inference.py`: 使用 LLM 进行推理
- `evaluate.py`: 评估脚本
- `README.md`: 本文件

## 成本估算

假设使用 GPT-4o-mini:
- 测试集大小: ~1000 样本
- 每个样本 tokens: ~1500 (输入) + 200 (输出)
- 价格: $0.15/1M input tokens, $0.6/1M output tokens
- 预估成本: ~$0.35

## 注意事项

1. API 调用可能超时,需要实现重试机制
2. 注意 rate limit,避免请求过快
3. 缓存 LLM 响应,避免重复调用
4. 对比 Level 1 的结果,分析优劣
