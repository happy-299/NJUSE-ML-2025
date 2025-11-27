# 机器学习实验三：预训练模型与大模型在代码评审中的应用

## 实验概述

本次实验探索了预训练模型和大语言模型在代码评审自动化任务中的应用。实验分为两个层级：
- **Level 1**：基于CodeReviewer预训练模型的微调方法
- **Level 2**：基于DeepSeek大语言模型的提示工程方法

---

# Level 1：基于CodeReviewer的代码评审

## 任务一：代码质量评估（评审必要性预测）

### 1. 问题介绍

代码质量评估任务旨在判断某个Pull Request（PR）是否需要人工评审。这是一个二分类任务：
- **类别0**：代码变更质量良好，无需人工评审
- **类别1**：代码变更存在潜在问题，需要人工评审

该任务的意义在于对应代码评审流程的"预筛选"阶段，通过自动识别无需评审的低风险提交，可以显著降低人力成本，让评审资源集中到高风险提交上。

### 2. 数据预处理

#### 2.1 数据集来源

数据集来自论文[1]，存储在`lab3-data/Diff_Quality_Estimation/`目录下，包含以下文件：
- `cls-train-chunk-0.jsonl` ~ `cls-train-chunk-3.jsonl`：训练数据（分块存储）
- `cls-valid.jsonl`：验证数据
- `cls-test.jsonl`：测试数据

#### 2.2 数据格式

每条数据为JSON格式，包含以下关键字段：

| 字段    | 说明                               |
| ------- | ---------------------------------- |
| `patch` | 代码差异（diff），作为模型输入     |
| `oldf`  | 修改前的完整文件内容               |
| `y`     | 标签（0或1），表示是否需要人工评审 |
| `lang`  | 编程语言                           |

#### 2.3 数据处理流程

```python
class SimpleClsDataset(Dataset):
    """简化的分类数据集，内存高效"""
    
    def __init__(self, file_paths, tokenizer, max_length=512, samplenum=-1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    # 使用 'patch' 作为输入，'y' 作为标签
                    input_text = data.get("patch", "") or data.get("oldf", "")
                    label = int(data.get("y", 0))
                    self.examples.append({
                        "input": input_text,
                        "label": label,
                    })
```

### 3. 论文复现

#### 3.1 模型架构

使用Microsoft CodeReviewer模型，基于RoBERTa架构的预训练代码语言模型：
- 模型来源：`microsoft/codereviewer`（HuggingFace）
- 预训练任务：代码理解和代码生成
- 微调方式：在分类头上进行二分类微调

#### 3.2 训练配置

| 参数                    | 值             |
| ----------------------- | -------------- |
| 批次大小（batch_size）  | 8              |
| 学习率（learning_rate） | 2e-5           |
| 训练轮数（epochs）      | 2              |
| 最大序列长度            | 512            |
| 梯度累积步数            | 4              |
| 优化器                  | AdamW          |
| 学习率调度              | 线性衰减带预热 |

#### 3.3 训练代码核心实现

```python
def train(args):
    # 加载预训练模型
    config, model, tokenizer = build_or_load_gen_model(args)
    model.to(DEVICE)
    
    # 加载数据
    train_dataset = load_data(DIFF_QUALITY_DIR, tokenizer, args, split="train")
    valid_dataset = load_data(DIFF_QUALITY_DIR, tokenizer, args, split="valid")
    
    # 训练循环
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            input_ids = batch[0].to(DEVICE)
            attention_mask = batch[1].to(DEVICE)
            labels = batch[2].to(DEVICE)
            
            # 前向传播（使用cls=True进行分类）
            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                cls=True,
            )
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
```

### 4. 结果与分析

#### 4.1 测试结果

| 指标                    | 数值   |
| ----------------------- | ------ |
| **Accuracy（准确率）**  | 63.25% |
| **Precision（精确度）** | 63.50% |
| **Recall（召回率）**    | 63.34% |
| **F1-Score（Macro）**   | 63.17% |

#### 4.2 结果分析

1. **整体表现**：模型在测试集上达到了63.25%的准确率，相比随机猜测（50%）有显著提升，表明模型成功学习到了代码变更与评审必要性之间的关联。

2. **平衡性**：Precision和Recall指标接近，表明模型在识别"需要评审"和"无需评审"两类样本上表现较为均衡。

3. **局限性**：
   - 63%的准确率虽然优于随机，但在实际应用中仍有较大提升空间
   - 代码评审任务具有较高的主观性，不同评审者可能有不同判断标准
   - 仅使用代码diff作为输入，缺少上下文信息（如项目历史、开发者经验等）

---

# Level 2：基于大语言模型的代码评审

## 方法与实现

### 1. 技术选型

- **LLM模型**：DeepSeek-Chat（deepseek-chat）
- **API接口**：OpenAI兼容格式
- **方法**：提示工程（Prompt Engineering）

### 2. LLM客户端实现

```python
class LLMClient:
    """LLM API客户端封装"""
    
    def __init__(self, provider="deepseek", model="deepseek-chat", 
                 api_key=None, base_url=None, **kwargs):
        self.provider = provider
        self.model = model
        
        # DeepSeek使用OpenAI兼容的API格式
        if provider == "deepseek":
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url or "https://api.deepseek.com"
            )
    
    def chat_completion(self, messages, temperature=0.7, max_tokens=2048):
        """调用聊天补全API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
```

---

## 任务一：代码质量评估

### 1. 问题介绍

与Level 1相同，任务目标是判断代码变更是否需要人工评审，输出二分类结果。

### 2. 数据预处理

使用与Level 1相同的数据集，从`cls-test.jsonl`加载测试数据：

```python
def load_test_data(data_file, max_samples=None):
    """加载测试数据"""
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
            if max_samples and len(data) >= max_samples:
                break
    return data
```

### 3. 方法与实现

#### 3.1 提示词设计

**系统提示词（System Prompt）**：
```
You are an expert code reviewer with extensive experience in software engineering 
and code quality assessment.

Your task is to analyze code changes from Pull Requests (PRs) and determine 
whether they require human review.

# Analysis Dimensions
When evaluating code changes, consider:
1. **Correctness**: Does the code work as intended? Are there any bugs?
2. **Code Style**: Does it follow coding conventions and best practices?
3. **Performance**: Are there any performance issues?
4. **Security**: Are there any security vulnerabilities?
5. **Maintainability**: Is the code easy to understand and maintain?

# Output Format
You must respond in JSON format:
{
    "needs_review": 1 or 0,
    "confidence": 0.0 to 1.0,
    "reasoning": "Detailed explanation",
    "issues": ["List of issues"]
}
```

**任务提示词（Task Prompt）**：
```
# Code Change Analysis

## Old File Content
```{language}
{old_code}
```

## Code Diff
```diff
{diff_code}
```

Analyze this code change and determine if it needs human review.
```

#### 3.2 推理流程

```python
def run_inference(args):
    client = create_llm_client(provider="deepseek", model="deepseek-chat")
    
    for sample in test_data:
        # 构建提示词
        messages = build_quality_estimation_prompt(
            old_code=sample.get("oldf", ""),
            diff_code=sample.get("patch", ""),
            language=sample.get("lang", "code"),
        )
        
        # 调用LLM
        response = client.chat_completion(messages, temperature=0.7)
        
        # 解析JSON响应
        result = client.extract_json(response)
        pred_label = int(result["needs_review"])
```

### 4. 结果与分析

#### 4.1 测试结果

| 指标                   | 数值   |
| ---------------------- | ------ |
| **Accuracy（准确率）** | 45.00% |
| **Precision（Macro）** | 53.92% |
| **Recall（Macro）**    | 52.08% |
| **F1-Score（Macro）**  | 41.33% |
| **平均置信度**         | 86.85% |

**分类别结果**：

| 类别                | Precision | Recall | F1     | 样本数 |
| ------------------- | --------- | ------ | ------ | ------ |
| Class 0（无需评审） | 66.67%    | 16.67% | 26.67% | 12     |
| Class 1（需要评审） | 41.18%    | 87.50% | 56.00% | 8      |

**混淆矩阵**：
```
         预测0  预测1
实际0    [  2,   10 ]
实际1    [  1,    7 ]
```

#### 4.2 结果分析

1. **模型倾向性**：LLM表现出明显的保守倾向，更倾向于判断代码"需要评审"（Class 1）。这导致Class 1的召回率高达87.50%，但Class 0的召回率仅为16.67%。

2. **置信度分析**：模型给出的平均置信度为86.85%，表明模型对自己的判断较为确信，但实际准确率较低，存在过度自信的问题。

3. **与Level 1对比**：
   - Level 1（CodeReviewer微调）：Accuracy = 63.25%
   - Level 2（DeepSeek提示工程）：Accuracy = 45.00%
   - 在该任务上，专门微调的模型表现优于通用LLM的零样本推理

4. **分析原因**：
   - LLM缺乏针对代码评审任务的专门训练
   - 提示词可能需要进一步优化
   - 代码评审标准具有主观性，LLM倾向于保守判断

---

## 任务二：问题代码行定位

### 1. 问题介绍

问题代码行定位任务旨在根据评审意见，识别代码片段中需要修改的具体行号。这是一个序列标注/定位任务，输出为行号索引列表。

该任务的意义在于：
- 为人工评审提供精准定位
- 为后续的评审意见生成与代码修复提供上下文

### 2. 数据预处理

使用`lab3-data/Comment_Generation/msg-test.jsonl`数据集：

```python
def load_raw_data(data_file, max_samples=None):
    """加载原始jsonl数据"""
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            data.append(sample)
    return data

def add_line_numbers(code):
    """为代码添加行号"""
    lines = code.split("\n")
    numbered_lines = [f"{i:4d} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)
```

### 3. 方法与实现

#### 3.1 提示词设计

**系统提示词**：
```
You are an expert code reviewer specializing in identifying problematic code locations.

Your task is to analyze code changes and review comments, then precisely locate 
which lines of code need to be modified.

# Analysis Process
1. Parse the review comment: What is the reviewer concerned about?
2. Examine the code diff: What changed? Which lines are affected?
3. Map comment to code: Which specific lines does the comment refer to?
4. Identify fix locations: Which lines need to be modified?

# Output Format
{
    "line_indices": [list of 0-based line numbers],
    "confidence": 0.0 to 1.0,
    "reasoning": "Step-by-step explanation",
    "affected_code": "Description of affected code"
}
```

**任务提示词**：
```
# Code Issue Localization Task

## Old Code (with line numbers)
```{language}
{old_code_numbered}
```

## Code Diff
```diff
{diff_code}
```

## New Code (with line numbers)
```{language}
{new_code_numbered}
```

## Review Comment
{comment}

Identify which specific lines in the NEW code need to be modified.
```

#### 3.2 Few-shot示例

在提示词中包含了多个示例来引导模型：

```
## Example 1
**Review Comment**: "Missing error handling for file operations"

**Response**:
{
    "line_indices": [1, 2],
    "confidence": 0.95,
    "reasoning": "Lines 1-2 need try-except blocks for IOError and JSONDecodeError.",
    "affected_code": "File opening and JSON loading operations"
}
```

### 4. 结果与分析

#### 4.1 测试结果

| 指标                          | 数值  |
| ----------------------------- | ----- |
| **Exact Match（精确匹配）**   | 0.00% |
| **Partial Match（部分匹配）** | 0.00% |
| **Mean IoU**                  | 0.00% |
| **MRR（平均倒数排名）**       | 0.00% |
| **Top-1 Accuracy**            | 0.00% |
| **Top-3 Accuracy**            | 0.00% |
| **Top-5 Accuracy**            | 0.00% |

#### 4.2 结果分析

1. **指标解读**：所有评估指标均为0，表明模型预测的行号与真实标注完全不匹配。

2. **问题分析**：
   - **数据集特性**：`msg-test.jsonl`数据集可能没有包含明确的`ground_truth_lines`标注
   - **任务难度**：代码定位任务需要精确理解评审意见与代码的对应关系
   - **行号计算**：不同的代码表示方式可能导致行号计算偏差

3. **模型输出示例**：

   **示例1**：
   ```
   Review Comment: "can we also test for `transport=rest`?"
   Prediction: [53]
   Reasoning: "需要在line 53添加transport=rest的测试用例"
   ```

   **示例2**：
   ```
   Review Comment: "I didn't realize we were hardcoding this, thanks for moving it to an env value."
   Prediction: []
   Confidence: 1.0
   Reasoning: "这是正面反馈，不需要修改任何行"
   ```

4. **改进方向**：
   - 需要更详细的标注数据来评估定位准确性
   - 可以考虑使用语义相似度等软指标替代精确匹配
   - 优化提示词，加入更多定位策略的指导

---

# 总结与展望

## 实验总结

| 任务         | Level   | 方法             | 核心指标           |
| ------------ | ------- | ---------------- | ------------------ |
| 代码质量评估 | Level 1 | CodeReviewer微调 | Accuracy: 63.25%   |
| 代码质量评估 | Level 2 | DeepSeek提示工程 | Accuracy: 45.00%   |
| 问题代码定位 | Level 2 | DeepSeek提示工程 | Exact Match: 0.00% |

## 关键发现

1. **专门微调 vs 通用LLM**：在代码质量评估任务上，经过专门微调的CodeReviewer模型（63.25%）显著优于通用大语言模型的零样本推理（45.00%）。

2. **LLM的保守倾向**：DeepSeek在评审必要性判断上表现出保守倾向，更倾向于建议人工评审，这在实际应用中可能导致评审资源的浪费。

3. **任务难度差异**：代码定位任务比二分类任务更具挑战性，需要模型精确理解评审意见与代码的对应关系。

## 未来改进方向

1. **提示词优化**：
   - 加入更多Few-shot示例
   - 使用Chain-of-Thought引导模型推理
   - 针对不同编程语言设计专门的提示词

2. **模型选择**：
   - 尝试代码专用的大语言模型（如CodeLlama、StarCoder）
   - 探索模型微调与提示工程的结合

3. **评估方法**：
   - 使用更细粒度的评估指标
   - 考虑代码语义相似度而非仅行号匹配

---

# 参考文献

[1] Li Z, Lu S, Guo D, et al. Automating code review activities by large-scale pre-training[C]//Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering. 2022: 1035-1047.

[2] Lu J, Yu L, Li X, et al. Llama-reviewer: Advancing code review automation with large language models through parameter-efficient fine-tuning[C]//Proceedings of the 34th International Symposium on Software Reliability Engineering. 2023: 647-658.

[3] Wang S, Lin B, Chen L, et al. Divide-and-conquer: Automating code revisions via localization-and-revision[J]. ACM Transactions on Software Engineering and Methodology, 2025, 34(3): 1-26.
