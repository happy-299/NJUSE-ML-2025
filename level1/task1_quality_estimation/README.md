# Task 1: 代码质量评估 (Level 1 - 论文复现)

## 任务描述

本任务使用 CodeReviewer 预训练模型复现论文中的代码质量评估(Diff Quality Estimation)任务。

### 问题定义

给定代码变更(old_file + diff_hunk),判断该变更是否需要人工评审(二分类任务)。

## 数据集

- **训练集**: `cls-train-chunk-0.jsonl` 到 `cls-train-chunk-3.jsonl`
- **验证集**: `cls-valid.jsonl`
- **测试集**: `cls-test.jsonl`

数据格式示例:
```json
{
    "old_hunk": "代码差异信息",
    "oldf": "旧文件内容",
    "label": 1  // 1: 需要评审, 0: 不需要评审
}
```

## 数据预处理

数据预处理步骤:
1. 读取 jsonl 格式数据
2. 提取 `old_hunk` 和 `oldf` 字段作为模型输入
3. 提取 `label` 字段作为标签
4. Tokenization (使用 CodeReviewer tokenizer)
5. 填充和截断 (max_source_length=512)

## 模型训练

### 预训练模型
- **模型名称**: microsoft/codereviewer
- **模型链接**: https://huggingface.co/microsoft/codereviewer

### 训练参数
```python
{
    "batch_size": 12,
    "learning_rate": 3e-4,
    "num_epochs": 30,
    "gradient_accumulation_steps": 3,
    "max_source_length": 512,
    "max_target_length": 128,
}
```

### 训练脚本

```bash
cd level1/task1_quality_estimation
bash sh/train.sh
```

## 模型测试

```bash
cd level1/task1_quality_estimation
bash sh/test.sh
```

## 评估指标

- Accuracy (准确率)
- Precision (精确度)
- Recall (召回率)
- F1-Score (Macro 平均)

## 预期结果

根据论文,预期结果大致为:
- Accuracy: ~85%
- F1-Score: ~85%

## 文件说明

- `train.py`: 训练脚本
- `test.py`: 测试脚本
- `inference.py`: 推理脚本
- `sh/train.sh`: 训练 bash 脚本
- `sh/test.sh`: 测试 bash 脚本

## 依赖安装

```bash
pip install torch transformers nltk
```

## 注意事项

1. 首次运行需要下载 CodeReviewer 预训练模型
2. 建议使用 GPU 进行训练
3. 训练时间较长,建议使用多 GPU 并行训练
4. 模型权重保存在 `level1/checkpoints/task1/` 目录
