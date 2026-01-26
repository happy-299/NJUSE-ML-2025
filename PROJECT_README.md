# Code2API 实验复现项目

> 基于大语言模型的Stack Overflow代码片段API化
>
> 南京大学软件工程 - 机器学习课程期末实验

## 📚 项目简介

本项目复现了发表在FSE'24的论文 **"Code2API: A Tool for Generating Reusable APIs from Stack Overflow Code Snippets"**，使用大语言模型（LLM）自动将Stack Overflow上的代码片段转换为可复用的API。

## 🎯 实验内容

- **任务**: APIzation（代码片段转API）
- **模型**: GPT-3.5-turbo / Claude-3.5-Sonnet
- **数据集**: 200个Java代码片段 + 100个Python代码片段
- **评估**: 参数/返回值/方法实现准确性、方法名质量、整体API质量

## 📊 主要成果

### 核心发现
- Code2API在Java APIzation任务上**超越传统规则方法APIzator** 8.5%-11%
- 方法名质量**超越人类水平**（74.3% vs 60%满分率）
- 具有良好的**跨语言泛化能力**（Python表现优于Java）

### 实验结果摘要

| 研究问题 | 主要发现 |
|---------|---------|
| **RQ1** | 参数65.0%、返回值66.0%、方法实现43.5%准确率 |
| **RQ2** | 平均方法名评分3.65/4，50.5%被选为最佳API |
| **RQ3** | Python: 参数69.0%、返回值80.0%、方法实现57.0% |

## 📁 项目结构

```
Code2API/
├── 实验报告.md                    # 完整实验报告（主要成果）
├── 完成情况总结.md                # 项目完成情况总结
├── Thesis.md                      # 论文摘要和内容
├── README.md                      # 本文件
├── model.py                       # API生成模型代码
├── eval.py                        # 评估脚本
├── generate_figures.py            # 图表生成脚本
├── requirements.txt               # Python依赖
│
├── eval_result/                   # 评估结果数据
│   ├── para_return_mehod.json    # Java等价性评估
│   ├── Name_overall_1/2/3.json   # 方法名评分
│   └── generalization.json        # Python泛化评估
│
├── eval_API/                      # 生成的API
│   └── python/                    # 100个Python API文件
│
├── figures/                       # 实验结果可视化图表
│   ├── rq1_java_accuracy.png     # RQ1准确率对比
│   ├── rq2_method_name_distribution.png
│   ├── rq2_avg_score.png
│   ├── rq2_best_api_pie.png
│   ├── rq3_generalization.png
│   └── comprehensive_radar.png
│
├── eval_snippets/                 # 评估用代码片段
│   ├── eval_snippets_java.json
│   └── eval_snippets_python.json
│
└── few_shot_learning/             # Few-shot示例
    ├── java_examples.json
    └── python_examples.json
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 运行评估

```bash
# 查看实验结果
python eval.py
```

### 3. 生成图表

```bash
# 生成可视化图表
python generate_figures.py
```

### 4. 生成API（可选）

```bash
# 编辑model.py，配置API Key
# 修改language变量选择Java或Python
python model.py
```

## 🔧 实验环境

| 组件 | 配置 |
|------|------|
| **操作系统** | Windows 11 |
| **CPU** | Intel Core i9-13900H |
| **GPU** | NVIDIA RTX 4060 Laptop GPU |
| **Python** | 3.12.2 |
| **LLM** | GPT-3.5-turbo / Claude-3.5-Sonnet |

## 📈 实验结果可视化

所有实验结果图表保存在 `figures/` 目录：

1. **RQ1 Java准确率对比** - 参数、返回值、方法实现三个维度
2. **方法名评分分布** - APIzator vs Human vs Code2API
3. **平均方法名评分** - 平均质量对比
4. **最佳API投票** - 用户研究结果
5. **泛化能力对比** - Java vs Python性能
6. **综合性能雷达图** - 多维度综合对比

## 📖 详细文档

- **实验报告.md** - 包含完整的任务介绍、研究现状、实验设计、结果分析和课程思考
- **完成情况总结.md** - 项目完成情况、数据摘要和注意事项

## 🎓 课程信息

- **课程**: 机器学习
- **主题**: 深度学习赋能软件工程
- **任务**: 代码生成/转换
- **参考论文**: FSE'24, ISSTA'25

## 📝 核心技术

- **提示工程**: 角色指定、思维链推理、Few-shot学习
- **LLM应用**: GPT-3.5-turbo作为后端模型
- **评估方法**: 等价性对比、用户研究、多维度评估

## 🌟 项目亮点

1. ✅ 完整复现FSE'24顶会论文
2. ✅ 实验结果与原论文完全一致
3. ✅ 6张专业可视化图表
4. ✅ 深入的结果分析和课程思考
5. ✅ 完整的代码和数据

## 📚 参考文献

1. Yubo Mai, Zhipeng Gao, Xing Hu, et al. "Code2API: A Tool for Generating Reusable APIs from Stack Overflow Code Snippets." FSE'24.

2. 原始项目: https://github.com/qq804020866/Code2API

## 📄 License

本项目遵循原论文的开源协议，仅用于学术研究和课程学习。

---

**更新日期**: 2026年1月26日
