# Level 1 Task 4: 代码修复生成

基于CodeReviewer模型的代码修复生成任务，根据代码审查意见自动生成修复后的代码。

## 📁 文件结构

```
task4_code_refinement/
├── basic_code_refinement_experiment.py    # 基础代码修复实验
└── README.md                              # 说明文档
```

## 🚀 快速使用

### 运行完整实验
```bash
python basic_code_refinement_experiment.py
```

## 📊 评估指标

- **Exact Match**: 精确匹配率
- **CodeBLEU**: 代码专用BLEU分数
- **按语言统计**: 各编程语言的准确率

## 💾 数据要求

数据文件放在 `../../data/raw/` 目录：
- `ref-train.jsonl` - 训练数据
- `ref-valid.jsonl` - 验证数据  
- `ref-test.jsonl` - 测试数据

## 🛠 技术架构

- **模型**: microsoft/codereviewer
- **框架**: PyTorch + Transformers
- **任务**: 序列到序列生成