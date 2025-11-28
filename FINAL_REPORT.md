# 最终实验报告（NJUSE-ML-2025）

**贡献说明**
- 肖继鹏：任务一（Level 1, 2）+ 任务二（Level 2）
- 张涛：任务三（Level 1, 2）
- 伍子科：任务四（Level 1, 2）
- 何浩川：任务全部（Level 3）

---

## Level 1

### 任务一：代码质量评估
- 问题介绍：给定代码变更（`old_file` + `diff_hunk`），判断该变更是否需要人工评审（二分类）。
- 数据预处理：提取`old_hunk`与`oldf`作为输入，`label`为目标；使用CodeReviewer分词器；`max_source_length=512`；填充与截断。
- 论文复现方法：
  - 预训练模型：`microsoft/codereviewer`
  - 训练参数：batch_size=12，learning_rate=3e-4，num_epochs=30，gradient_accumulation_steps=3，max_source_length=512，max_target_length=128。
  - 训练/测试脚本位置：`level1/task1_quality_estimation/sh/train.sh`、`level1/task1_quality_estimation/sh/test.sh`
- 结果与分析：
  - 来自附件文档 `level1_task1-level2_task1-2-实验报告.md` 的测试结果：
    - Accuracy（准确率）：63.25%
    - Precision（精确度）：63.50%
    - Recall（召回率）：63.34%
    - F1-Score（Macro）：63.17%
  - 分析与结论：在测试集上较随机基线显著提升，Precision/Recall较为平衡；受限于仅使用代码diff作为输入，仍有提升空间（项目历史、开发者上下文等）。

### 任务三：评审意见生成
- 问题介绍：根据Diff Hunk自动生成自然语言评审意见。
- 数据预处理：
  - 将`+/-/context`标准化为`[ADD]/[DEL]/[KEEP]`特殊token。
  - 源序列截断至300，目标序列截断至128。
- 论文复现：
  - 模型：CodeT5-base（约220M参数）。
  - 训练：在CPU环境下采用随机采样训练集3,000条、1个epoch；AdamW+LinearWarmup；Batch=2。
  - 推理：Beam Size=1（Greedy）以加速。
- 结果与分析（来自项目输出文件）：
  - 指标（`outputs/level1/task3_formal/Level1-Task3 实验报告.md`）：
    - BLEU-4: 0.00
    - ROUGE-L: 1.41
    - BERTScore F1: 81.11
  - 结论：在受限算力与缩减数据条件下，语义相似性较好（BERTScore高），但n-gram精确匹配（BLEU）极低，符合任务的高多样性与困难度预期。

### 任务四：修复代码生成
- 问题介绍：依据评审意见与代码上下文，直接生成修复后的代码。
- 数据预处理与流程：搭建数据加载→模型推理→结果评估的完整pipeline，支持多语言样本。
- 论文复现：
  - 预训练模型：`microsoft/codereviewer`
  - 技术栈：Python 3.12 + PyTorch 2.9.1 + Transformers 4.57.1
- 结果与分析（来自项目输出文件）：
  - 小规模实验（`outputs/level1/task4/EXPERIMENT_SUMMARY.md`）：
    - 测试样本：50
    - 精确匹配率：0.00%（0/50）
    - 语言覆盖：Go/Java/Python/JavaScript/C/Ruby/PHP/C++（各语言均0）
    - 关键发现：模型偏向生成评审意见格式而非修复代码；需微调与提示工程优化。
  - 大规模实验（`outputs/level1/task4/LARGE_SCALE_EXPERIMENT_SUMMARY.md`）：
    - 测试规模：13,104样本，覆盖9种语言
    - 精确匹配率：0.0000%（0/13,104）
    - 科学意义：验证任务复杂性与建立评估基准，指明需专门微调与更强模型。

---

## Level 2（在 Level 1 基础上额外提交）

### 方法与实现（通用）
- LLM模型与提示设计：
  - 继续基于`microsoft/codereviewer`与CodeT5等代码预训练模型。
  - 结构化提示模板（以Task4为例），显式指令“仅输出修复后代码”，提供语言上下文并设定输出段落标记（如`Fixed Code:`），引入否定指令约束输出格式。
- 生成参数优化：
  - 相较Level 1，Level 2尝试`num_beams=4`、`temperature=0.7`、`do_sample=True`、`repetition_penalty=1.2`等，以平衡探索与一致性。
- 后处理机制：
  - 提取标记段内容、过滤`<msg>`评论样式、清理无关文本，提高输出格式可用性。

### 任务一/三/四：将“论文复现”替换为“方法与实现”
- 延续上述提示工程、参数调优与后处理的设计；在各任务中针对输入结构（分类/生成/代码修复）进行适配。

### 任务二：问题定位
- 问题介绍：根据评审意见与代码变更，自动定位问题代码片段或函数范围。
- 数据预处理：解析diff与文件内容，统一token化，控制最大输入长度并保留结构标记。
- 方法与实现：
  - 使用LLM进行代码语义对齐与片段匹配；提示中包含“语言”“上下文”“目标定位类型（行/函数/文件）”。
  - 输出后处理对齐到具体位置标识（如起止行号或函数名）。
- 结果与分析（来自附件文档 `level1_task1-level2_task1-2-实验报告.md`）：
  - 指标：Exact Match 0.00%、Partial Match 0.00%、Mean IoU 0.00%、MRR 0.00%、Top-1/3/5 Accuracy 0.00%。
  - 结论：当前评估指标均为0，提示对行级定位的严格评估尚未达到；建议引入更详细标注、软指标（语义相似度）、以及更强的定位策略与工具辅助。

### Level 2 任务四（高级代码修复）结果
- 来自`outputs/level2/task4/LEVEL2_SUMMARY.md`：
  - 核心结果：总样本数200；精确匹配0；准确率0.0000%；与Level 1相比无显著提升。
  - 语言表现：各语言均为0。
  - 方法与实现：结构化提示、参数调优（beam/temperature/采样）、后处理（过滤评论样式）。
  - 分析结论：预训练偏差强烈（更擅评审意见生成），仅靠提示工程难以扭转；需要深度微调与更强模型。

### Level 2 任务一（LLM 提示工程）对比结果
- 来自附件文档 `level1_task1-level2_task1-2-实验报告.md`：
  - 指标（DeepSeek-Chat 零样本）：Accuracy 45.00%、Precision（Macro）53.92%、Recall（Macro）52.08%、F1（Macro）41.33%、平均置信度 86.85%。
  - 类别分析：Class 1（需要评审）召回率显著较高（87.50%），Class 0召回率较低（16.67%），呈保守倾向。
  - 与Level 1微调对比：CodeReviewer微调（Accuracy 63.25%）优于LLM零样本（Accuracy 45.00%），显示专门微调对该任务更有效。

---

## Level 3

### 方法与实现（GraphRAG / AI Agent）
- 整体流程与步骤：
  1. 数据装载与任务配置：`level3/agent/data_loader.py`、`level3/agent/config.py`
  2. LLM工厂与客户端封装：`level3/agent/llm_factory.py`、`level3/agent/prompt_utils.py`
  3. 工作流编排：
     - 全流程管线：`level3/agent/workflows/full_pipeline_workflow.py`
     - 评审工作流：`level3/agent/workflows/review_workflow.py`
     - 任务1专用：`level3/agent/workflows/task1_workflow.py`
  4. 工具调用：`level3/agent/tools/code_analyzer.py`（代码结构/依赖分析）
  5. 指标与度量：`level3/agent/metrics.py`
  6. 入口与状态管理：`level3/agent/main.py`、`level3/agent/state.py`
- 说明：上述Agent架构支持将四个任务整合到一个多步推理与工具链调用的管线中；可扩展接入GraphRAG以构建代码知识图、进行上下文检索与证据汇聚，提升多文件场景下的理解与定位能力。

### 结果与分析
- 性能对比与改进效果：基于Level 1/2的经验，Agent与GraphRAG的潜在改进点包括：
  - 多步计划与工具调用可提升任务二（定位）的结构化准确性。
  - 通过检索增强上下文，任务三（评审意见）在语义相关性上可能提升（例如BERTScore）。
  - 对任务四（代码修复），若引入执行反馈与语法/静态分析循环，有望提升可执行性与功能等效性（即便精确匹配仍难）。

---

## 总结与建议
- 任务难度：代码修复生成在小规模与大规模实验中均表现出极高难度（精确匹配≈0），验证了研究问题的挑战性。
- 方法价值：Level 2的提示工程与参数调优建立了良好的技术基线；Level 3的Agent架构为进一步改进提供了清晰路径。
- 后续方向：
  - 深度微调：在特定数据集上进行监督微调或指令调优。
  - 评估扩展：引入语义相似度、功能等效性、可执行性验证等更契合代码场景的指标。
  - 多模态融合：结合AST、类型系统与静态/动态分析工具。
  - 人机协作：设计交互式修复与审查工具，将LLM与工程师经验结合。

---

## 引用与数据来源
- `outputs/level1/task4/EXPERIMENT_SUMMARY.md`
- `outputs/level1/task4/LARGE_SCALE_EXPERIMENT_SUMMARY.md`
- `outputs/level1/task3_formal/Level1-Task3 实验报告.md`
- `outputs/level2/task4/LEVEL2_SUMMARY.md`
- 附件：`level1_task1-level2_task1-2-实验报告.md`
- `level1/task1_quality_estimation/README.md`
- `level3/agent/*`（方法与实现架构说明）