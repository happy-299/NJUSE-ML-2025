# Task 4: 修复代码生成 (Level 1 - 论文复现)

## 任务描述

本任务使用 CodeReviewer 预训练模型复现论文中的修复代码生成(Code Refinement)任务。

### 问题定义

给定代码差异(old_hunk + diff_hunk)和评审意见(comment),生成修复后的代码(refined code)。这是一个代码生成任务。

## 数据集

- **训练集**: `ref-train.jsonl`
- **验证集**: `ref-valid.jsonl` 
- **测试集**: `ref-test.jsonl`

数据格式示例:
```json
{
    "old_hunk": "原始代码差异",
    "oldf": "旧文件内容", 
    "hunk": "代码变更",
    "comment": "评审意见",
    "new": "修复后的代码",
    "old": "原始代码片段",
    "lang": "编程语言"
}
```

## 数据预处理

数据预处理步骤:
1. 读取 jsonl 格式数据
2. 构造输入: old_hunk + comment 作为源序列
3. 构造输出: new 字段作为目标序列
4. Tokenization (使用 CodeReviewer tokenizer)
5. 填充和截断 (max_source_length=512, max_target_length=128)

## 模型训练

### 预训练模型
- **模型名称**: microsoft/codereviewer
- **模型链接**: https://huggingface.co/microsoft/codereviewer

### 训练参数
```python
{
    "batch_size": 8,
    "learning_rate": 3e-4,
    "num_epochs": 10,
    "gradient_accumulation_steps": 4,
    "max_source_length": 512,
    "max_target_length": 128,
}
```

### 训练脚本

```bash
cd level1/task4_code_refinement
bash sh/train.sh
```

## 模型测试

```bash
cd level1/task4_code_refinement
bash sh/test.sh
```

## 推理

```bash
cd level1/task4_code_refinement
python inference.py --model_path checkpoints/task4-best --input_file test_input.jsonl --output_file predictions.jsonl
```

## 评估指标

- **Exact Match (EM)**: 完全匹配准确率
- **CodeBLEU**: 代码专用的BLEU指标,结合AST结构和数据流
- **BLEU-4**: 标准BLEU-4分数
- **ROUGE-L**: 最长公共子序列指标

## 预期结果

根据论文,预期结果大致为:
- **Exact Match**: ~15-25%
- **CodeBLEU**: ~25-35%
- **BLEU-4**: ~20-30%

## 文件说明

- `train.py`: 训练脚本
- `test.py`: 测试和评估脚本
- `inference.py`: 推理脚本
- `sh/train.sh`: 训练 bash 脚本
- `sh/test.sh`: 测试 bash 脚本
- `sh/inference.sh`: 推理 bash 脚本

## 依赖安装

```bash
pip install torch transformers nltk tree-sitter
# 安装 CodeBLEU 评估工具
pip install codebleu
```

## 注意事项

1. **数据大小**: 修复代码生成数据集较大,建议使用数据采样进行快速实验
2. **生成长度**: 目标代码长度变化较大,需要适当调整 max_target_length
3. **评估复杂**: CodeBLEU 需要解析代码AST,对不同编程语言有不同要求
4. **模型权重**: 保存在 `level1/checkpoints/task4/` 目录
5. **内存使用**: 代码生成任务显存需求较大,建议使用较小的 batch_size

## 数据采样策略

由于完整数据集较大,建议按以下策略采样:
- 训练集: 随机采样 10K-50K 样本
- 验证集: 全量使用或采样 5K 样本
- 测试集: 全量使用进行最终评估

## 编程语言分布

数据集包含多种编程语言,主要包括:
- Java, Python, C++, JavaScript, C#, Ruby 等
- 建议按语言分层抽样以保持分布一致性