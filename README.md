# NJUSE-ML-2025 Lab3: 代码评审自动化实验

本项目为南京大学软件学院 2025 年机器学习课程实验 3：代码评审自动化实验的项目仓库。

## 实验概述

本实验探索预训练模型和大模型在代码评审中的应用，包含四个核心任务：
1. **任务一**：代码质量评估（评审必要性预测，二分类任务）
2. **任务二**：问题代码行定位
3. **任务三**：评审意见生成
4. **任务四**：修复代码生成

实验分为三个难度层级：
- **Level 1**：使用 CodeReviewer 论文的开源代码和 checkpoint 完成任务一、三、四
- **Level 2**：使用提示工程(Prompt Engineering)或微调技术完成全部四个任务
- **Level 3**：使用 GraphRAG 或 AI Agent 技术完成全部四个任务

## 项目目录结构

```
NJUSE-ML-2025/
├── README.md                   # 项目说明文档
├── requirements.txt            # 项目依赖
├── config.py                   # 全局配置文件
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据(从 lab3-data 复制或软链接)
│   │   ├── Diff_Quality_Estimation/   # 任务一数据集
│   │   ├── Comment_Generation/        # 任务三数据集
│   │   └── Code_Refinement/           # 任务四数据集
│   ├── processed/              # 预处理后的数据
│   └── preprocess.py           # 数据预处理脚本
│
├── level1/                     # Level 1: 论文复现
│   ├── task1_quality_estimation/     # 任务一：代码质量评估
│   │   ├── train.py           # 训练脚本
│   │   ├── test.py            # 测试脚本
│   │   ├── inference.py       # 推理脚本
│   │   └── README.md          # 任务说明
│   ├── task3_comment_generation/     # 任务三：评审意见生成
│   │   └── README.md
│   ├── task4_code_refinement/        # 任务四：修复代码生成
│   │   └── README.md
│   └── checkpoints/           # Level 1 模型权重
│
├── level2/                     # Level 2: 提示工程
│   ├── task1_quality_estimation/     # 任务一：代码质量评估
│   │   ├── prompts/           # 提示词模板
│   │   │   ├── system_prompt.txt
│   │   │   └── task_prompt.txt
│   │   ├── inference.py       # 使用 LLM 进行推理
│   │   ├── evaluate.py        # 评估脚本
│   │   └── README.md
│   ├── task2_code_localization/      # 任务二：问题代码定位
│   │   ├── prompts/
│   │   ├── inference.py
│   │   ├── evaluate.py
│   │   └── README.md
│   ├── task3_comment_generation/     # 任务三：评审意见生成
│   │   └── README.md
│   ├── task4_code_refinement/        # 任务四：修复代码生成
│   │   └── README.md
│   └── shared/                # Level 2 共享资源
│       ├── llm_client.py      # LLM API 调用封装
│       └── prompt_utils.py    # 提示词工具函数
│
├── level3/                     # Level 3: AI Agent
│   ├── agent/                 # AI Agent 核心代码
│   │   ├── agent_base.py      # Agent 基类
│   │   ├── tools.py           # Agent 工具集
│   │   └── workflow.py        # Agent 工作流
│   ├── prompts/               # Agent 提示词(可复用 level2 的提示词)
│   ├── run_agent.py           # Agent 运行脚本
│   └── README.md
│
├── utils/                      # 通用工具函数
│   ├── metrics.py             # 评估指标计算
│   ├── visualization.py       # 可视化工具
│   └── data_utils.py          # 数据处理工具
│
├── outputs/                    # 输出目录
│   ├── level1/                # Level 1 实验结果
│   ├── level2/                # Level 2 实验结果
│   └── level3/                # Level 3 实验结果
│
└── notebooks/                  # Jupyter Notebooks(可选)
    ├── data_analysis.ipynb    # 数据分析
    └── result_visualization.ipynb  # 结果可视化
```

## 任务分工说明

每位组员根据自己的分工在对应的 `level*/task*/` 目录下完成任务，具体包括：

1. **数据预处理**：在 `data/preprocess.py` 中实现或调用预处理函数
2. **模型实现/提示词设计**：
   - Level 1: 在 `level1/task*/` 下复现论文代码
   - Level 2: 在 `level2/task*/prompts/` 下设计提示词
   - Level 3: 在 `level3/` 下实现 AI Agent，可复用 Level 2 的提示词
3. **实验运行**：使用提供的训练/推理脚本运行实验
4. **结果分析**：将结果保存到 `outputs/` 对应目录，并进行分析

## 环境配置

### 基础依赖

```bash
# 创建虚拟环境
conda create -n code-review python=3.8
conda activate code-review

# 安装依赖
pip install -r requirements.txt
```

### Level 1 依赖

```bash
# PyTorch (根据 CUDA 版本调整)
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# Transformers
pip install transformers

# NLTK
conda install nltk
```

### Level 2 依赖

```bash
# OpenAI API (如使用 GPT 系列)
pip install openai

# 或其他 LLM API 客户端
# pip install anthropic  # Claude
# pip install google-generativeai  # Gemini
```

### Level 3 依赖

```bash
# LangChain (用于构建 AI Agent)
pip install langchain langchain-openai

# 或其他 Agent 框架
# pip install autogen
```

## 快速开始

### Level 1: 论文复现

```bash
# 任务一：代码质量评估
cd level1/task1_quality_estimation
python train.py --model_path <预训练模型路径>
python test.py --model_path <训练好的模型路径>
```

### Level 2: 提示工程

```bash
# 任务一：代码质量评估
cd level2/task1_quality_estimation
python inference.py --api_key <your_api_key>
python evaluate.py

# 任务二：问题代码定位
cd level2/task2_code_localization
python inference.py --api_key <your_api_key>
python evaluate.py
```

### Level 3: AI Agent

```bash
cd level3
python run_agent.py --tasks all
```

## 数据集说明

数据集来自论文 [CodeReviewer](https://arxiv.org/abs/2203.09095)，从 GitHub 爬取多个开源项目的 PR 数据。

- **下载链接**：https://zenodo.org/records/6900648
- **数据格式**：参见 `data/README.md`

## 评价指标

- **任务一**：Accuracy, Precision, Recall, F1-Score (Macro)
- **任务二**：Accuracy, Precision, Recall, F1-Score, MRR (Mean Reciprocal Rank)
- **任务三**：BLEU-4, ROUGE-L, BERTScore
- **任务四**：Exact Match, CodeBLEU

详细计算方法见 `utils/metrics.py`。

## 注意事项

1. **数据路径**：请在 `config.py` 中配置数据集路径
2. **API 密钥**：Level 2/3 需要配置 LLM API 密钥，建议使用环境变量
3. **模型权重**：Level 1 需要下载 CodeReviewer 预训练模型
4. **Git 协作**：
   - 不要提交大文件(数据集、模型权重)到 Git
   - 在 `.gitignore` 中添加 `data/raw/`, `checkpoints/`, `*.pt`, `*.pth`
   - 每个任务在独立分支开发，完成后 PR 到 main

## 参考资料

- [CodeReviewer 论文](https://arxiv.org/abs/2203.09095)
- [CodeReviewer GitHub](https://github.com/microsoft/CodeBERT/tree/master/CodeReviewer)
- [CodeReviewer 模型](https://huggingface.co/microsoft/codereviewer)

## 贡献者

- 组员1：任务一(Level 1, 2) + 任务二(Level 2)
- 组员2：任务三(Level 1, 2)
- 组员3：任务四(Level 1, 2)
- 组员4：任务全部(Level 3)

(请根据实际分工更新)
