# 快速开始指南

本文档帮助你快速上手实验任务。

## 环境准备

### 1. 创建虚拟环境

```bash
conda create -n code-review python=3.8
conda activate code-review
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载数据集

从 https://zenodo.org/records/6900648 下载数据集,解压到 `data/raw/` 目录:

```
data/raw/
├── Diff_Quality_Estimation/
├── Comment_Generation/
└── Code_Refinement/
```

### 4. 配置 API 密钥 (Level 2/3 需要)

```bash
# Windows (cmd)
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

或在 `config.py` 中直接配置。

---

## Level 1: 论文复现

### 任务一:代码质量评估

#### 1. 下载预训练模型

模型会自动从 Hugging Face 下载,或手动下载:
- https://huggingface.co/microsoft/codereviewer

#### 2. 训练模型

```bash
cd level1/task1_quality_estimation
python train.py
```

或使用 bash 脚本 (Linux/Mac):
```bash
cd level1/task1_quality_estimation
bash sh/train.sh
```

#### 3. 测试模型

```bash
python test.py --model_path ../../level1/checkpoints/task1/checkpoint-best
```

#### 4. 查看结果

结果保存在 `outputs/level1/task1/test_results.json`

---

## Level 2: 提示工程

### 任务一:代码质量评估

#### 1. 配置 API 密钥

确保已设置 `OPENAI_API_KEY` 环境变量。

#### 2. 运行推理

```bash
cd level2/task1_quality_estimation
python inference.py --model gpt-4o-mini --max_samples 100
```

参数说明:
- `--model`: LLM 模型名称
- `--max_samples`: 限制样本数量(用于测试)
- `--temperature`: 温度参数(默认 0.7)

#### 3. 评估结果

```bash
python evaluate.py --predictions_file predictions.json --analyze_errors
```

#### 4. 查看结果

结果保存在 `outputs/level2/task1/`:
- `predictions.json`: 预测结果
- `evaluation_results.json`: 评估指标
- `error_analysis.json`: 错误分析

### 任务二:问题代码定位

#### 1. 准备数据

```bash
cd level2/task2_code_localization
python prepare_data.py --source comment_generation --max_samples 100
```

#### 2. 运行推理

```bash
python inference.py --model gpt-4o-mini
```

#### 3. 评估结果

```bash
python evaluate.py --predictions_file predictions.json --analyze_errors
```

---

## 常见问题

### Q1: 数据集在哪里?

A: 需要从 https://zenodo.org/records/6900648 下载,解压到 `data/raw/` 目录。

### Q2: 如何修改提示词?

A: 编辑 `level2/task*/prompts/` 目录下的 `.txt` 文件。

### Q3: API 调用超时怎么办?

A: 在推理脚本中增加 `--retry_attempts` 参数,或减少 `--max_samples`。

### Q4: 如何切换 LLM 模型?

A: 使用 `--provider` 和 `--model` 参数,例如:
```bash
python inference.py --provider anthropic --model claude-3-opus-20240229
```

### Q5: 内存不足怎么办?

A: Level 1 训练时减少 `--batch_size` 参数。

---

## 文件结构速查

- `level1/task1_quality_estimation/`: Level 1 任务一代码
  - `train.py`: 训练脚本
  - `test.py`: 测试脚本
- `level2/task1_quality_estimation/`: Level 2 任务一代码
  - `prompts/`: 提示词模板
  - `inference.py`: 推理脚本
  - `evaluate.py`: 评估脚本
- `level2/task2_code_localization/`: Level 2 任务二代码
  - `prompts/`: 提示词模板
  - `prepare_data.py`: 数据准备
  - `inference.py`: 推理脚本
  - `evaluate.py`: 评估脚本
- `level2/shared/`: Level 2 共享模块
  - `llm_client.py`: LLM API 客户端
  - `prompt_utils.py`: 提示词工具
- `utils/`: 通用工具
  - `metrics.py`: 评估指标
- `config.py`: 全局配置
- `requirements.txt`: 依赖列表

---

## 实验报告建议

### Level 1 任务一

1. **问题介绍**: 代码质量评估任务定义
2. **数据预处理**: 数据格式、统计信息
3. **论文复现**: 模型架构、训练参数、训练过程
4. **结果与分析**: 
   - 在测试集上的指标 (Accuracy, Precision, Recall, F1)
   - 与论文结果对比
   - 错误分析

### Level 2 任务一

1. **问题介绍**: 同 Level 1
2. **数据预处理**: 同 Level 1
3. **方法与实现**: 
   - 使用的 LLM 模型
   - 提示词设计思路 (System Prompt + Task Prompt)
   - Few-shot 示例选择
4. **结果与分析**:
   - 在测试集上的指标
   - 与 Level 1 对比
   - 提示词优化过程
   - 成本分析

### Level 2 任务二

1. **问题介绍**: 问题定位任务定义
2. **数据预处理**: 数据准备过程、标注方法
3. **方法与实现**:
   - 提示词设计 (如何引导 LLM 定位代码行)
   - 行号处理策略
4. **结果与分析**:
   - Exact Match, MRR, Top-K Accuracy
   - 典型成功/失败案例分析

---

## 联系与协作

- 每个任务独立开发,避免冲突
- 提交前 pull 最新代码
- 不要提交大文件 (数据集、模型权重)
- 及时同步进度

祝实验顺利! 🎉
