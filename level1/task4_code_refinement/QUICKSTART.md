# Level 1 Task 4: 修复代码生成 - 快速开始指南

## 概述

任务四实现了基于 CodeReviewer 模型的代码修复生成功能。给定原始代码差异和评审意见，模型可以生成修复后的代码。

## 项目结构

```
level1/task4_code_refinement/
├── README.md                # 详细文档
├── train.py                 # 训练脚本
├── test.py                  # 测试和评估脚本
├── inference.py             # 推理脚本
├── preprocess_data.py       # 数据预处理脚本
└── sh/                      # Shell脚本
    ├── train.sh             # 训练
    ├── test.sh              # 测试
    ├── inference.sh         # 批量推理
    ├── interactive.sh       # 交互式推理
    └── quick_test.sh        # 快速测试
```

## 快速开始

### 1. 环境配置

```bash
# 安装基础依赖
pip install torch transformers nltk

# 安装评估工具
pip install rouge codebleu bert-score

# 安装可视化工具 (可选)
pip install matplotlib seaborn
```

### 2. 数据预处理 (可选)

```bash
# 分析数据集
cd level1/task4_code_refinement
python preprocess_data.py --visualize --create_samples

# 查看数据统计信息
python preprocess_data.py --data_dir ../../data/raw
```

### 3. 模型训练

```bash
# 使用默认参数训练
bash sh/train.sh

# 或者自定义参数
python train.py \
  --model_name_or_path microsoft/codereviewer \
  --batch_size 8 \
  --learning_rate 3e-4 \
  --num_epochs 10 \
  --max_train_samples 50000
```

### 4. 模型测试

```bash
# 快速测试 (小样本)
bash sh/quick_test.sh

# 完整测试
bash sh/test.sh

# 自定义测试
python test.py \
  --model_path level1/checkpoints/task4/checkpoint-best \
  --max_test_samples 1000
```

### 5. 模型推理

#### 批量推理
```bash
# 使用测试数据
bash sh/inference.sh

# 使用自定义数据
bash sh/inference.sh input.jsonl output.jsonl
```

#### 交互式推理
```bash
# 启动交互模式
bash sh/interactive.sh
```

在交互模式下，你可以输入代码和评审意见：
```
Old code (end with '###'):
def add(a, b):
    return a + b
###

Comment: Add type hints and validation

Refined Code:
def add(a: int, b: int) -> int:
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("Arguments must be integers")
    return a + b
```

## 数据格式

### 输入数据格式 (JSONL)
```json
{
  "old_hunk": "原始代码差异",
  "comment": "评审意见",
  "new": "修复后的代码",
  "lang": "编程语言",
  "repo": "仓库名称"
}
```

### 输出结果格式
```json
{
  "old_hunk": "原始代码差异",
  "comment": "评审意见", 
  "refined_code": "生成的修复代码",
  "original_id": 0,
  "language": "java",
  "reference": "真实的修复代码"
}
```

## 评估指标

- **Exact Match (EM)**: 完全匹配准确率
- **CodeBLEU**: 代码专用BLEU指标，结合AST结构
- **BLEU-4**: 标准4-gram BLEU分数
- **ROUGE-L**: 最长公共子序列指标

## 预期性能

根据论文和实验结果：
- **Exact Match**: 15-25%
- **CodeBLEU**: 25-35%
- **BLEU-4**: 20-30%

## 配置说明

### 训练配置
```python
{
    "batch_size": 8,              # 批次大小
    "learning_rate": 3e-4,        # 学习率
    "num_epochs": 10,             # 训练轮数
    "max_source_length": 512,     # 输入最大长度
    "max_target_length": 128,     # 输出最大长度
    "max_train_samples": 50000,   # 训练样本数限制
    "max_valid_samples": 5000     # 验证样本数限制
}
```

### 生成配置
```python
{
    "num_beams": 5,               # Beam搜索宽度
    "temperature": 1.0,           # 生成温度
    "do_sample": False            # 是否采样
}
```

## 故障排除

### 1. 内存不足
```bash
# 减少批次大小
python train.py --batch_size 4 --gradient_accumulation_steps 8

# 使用数据采样
python train.py --max_train_samples 10000
```

### 2. 模型不存在
```bash
# 检查模型路径
ls -la level1/checkpoints/task4/

# 重新训练
bash sh/train.sh
```

### 3. 数据格式错误
```bash
# 检查数据格式
python preprocess_data.py --data_dir ../../data/raw

# 查看样本数据
head -3 ../../data/raw/ref-train.jsonl
```

### 4. 依赖问题
```bash
# 安装缺失的包
pip install transformers torch nltk rouge codebleu bert-score

# 下载NLTK数据
python -c "import nltk; nltk.download('punkt')"
```

## 高级用法

### 1. 自定义数据训练
```python
# 准备你的数据 (JSONL格式)
# 每行包含: old_hunk, comment, new, lang

# 修改配置
python train.py \
  --input_file your_data.jsonl \
  --output_dir your_model_output \
  --max_epochs 20
```

### 2. 多语言支持
```python
# 训练时按语言分层采样
python preprocess_data.py \
  --create_samples \
  --sample_strategy balanced \
  --train_sample_size 30000
```

### 3. 模型微调
```python
# 从已有模型继续训练
python train.py \
  --model_name_or_path level1/checkpoints/task4/checkpoint-best \
  --learning_rate 1e-4 \
  --num_epochs 5
```

### 4. 批量评估
```python
# 评估多个检查点
for checkpoint in level1/checkpoints/task4/checkpoint-*; do
  echo "Evaluating $checkpoint"
  python test.py --model_path "$checkpoint"
done
```

## 实验报告撰写建议

### 问题介绍
- 代码修复生成的背景和重要性
- 任务定义和挑战
- CodeReviewer模型介绍

### 数据预处理
- 数据集统计分析
- 数据清洗和格式化
- 训练/验证/测试划分

### 论文复现
- CodeReviewer模型架构
- 训练过程和参数设置
- 收敛曲线和训练日志

### 结果与分析
- 各项评估指标结果
- 与论文结果对比
- 错误分析和案例研究
- 不同编程语言的性能差异

## 参考资料

- [CodeReviewer 论文](https://arxiv.org/abs/2203.09095)
- [CodeReviewer GitHub](https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer)
- [CodeBLEU 评估工具](https://github.com/microsoft/CodeXGLUE)
- [Transformers 文档](https://huggingface.co/docs/transformers)

## 联系方式

如有问题，请查看：
1. README.md 详细文档
2. 项目issues
3. 代码注释和日志输出