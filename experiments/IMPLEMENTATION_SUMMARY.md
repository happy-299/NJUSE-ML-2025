# Level2 实现总结

## 📋 已实现的功能

### 1. 模型组件消融实验 ✅

**文件**：`experiments/ablation_study.py`

**功能**：
- 支持 6 种消融变体（full, no_shared, single_tower, no_expert, shallow, no_dropout）
- 自动训练所有变体并保存结果
- 生成汇总报告（JSON 格式）
- 支持自定义变体选择

**使用**：
```powershell
python experiments\ablation_study.py --config configs\django_mmoe.yaml --output outputs\ablation
```

### 2. 泛化性实验（跨项目预测）✅

**文件**：`experiments/cross_project.py`

**功能**：
- 项目内基线实验（Within-Project）
- 跨项目迁移实验（Cross-Project）
- 混合训练实验（Mixed-Training）
- 支持多项目数据加载与对齐
- 自动生成泛化性汇总报告

**使用**：
```powershell
python experiments\cross_project.py --config configs\django_shared.yaml --output outputs\cross_project --projects django
```

### 3. 假设检验 ✅

**文件**：`experiments/hypothesis_testing.py`

**功能**：
- Friedman test（非参数主检验）
- Nemenyi post-hoc test（成对比较）
- Kruskal-Wallis test + Dunn's test（替代方案）
- 自动生成统计检验报告
- 可视化：性能对比图、秩次图、成对比较热图

**使用**：
```powershell
python experiments\hypothesis_testing.py --summaries outputs\ablation\ablation_summary.json --output outputs\hypothesis_testing
```

### 4. 辅助工具 ✅

#### 一键运行脚本
**文件**：`experiments/run_all.py`

自动运行所有实验（消融、泛化性），并提示假设检验命令。

#### 结果可视化
**文件**：`experiments/visualize_experiments.py`

生成：
- 消融实验对比图
- 泛化性实验对比图
- 综合结果表格（Markdown + CSV）

#### 单元测试
**文件**：`experiments/test_experiments.py`

验证假设检验工具、配置定义等基本功能。

### 5. 文档 ✅

- **README.md**：更新项目主文档，添加 level2 说明
- **experiments/README.md**：详细的实验设计与方法文档
- **experiments/QUICKSTART.md**：快速入门指南
- **REPORT.md**：完整实验报告（已添加 level2 章节 6-10）

### 6. 配置文件 ✅

- `configs/ablation_baseline.yaml`：Shared-Bottom 消融基线
- `configs/ablation_mmoe.yaml`：MMoE 消融基线

---

## 📊 REPORT.md 更新内容

已在 `REPORT.md` 中添加以下章节：

### 第 6 节：特征工程（占位）
- 说明：由负责特征工程的团队成员补充
- 内容：特征提取、选择、变换、重要性分析

### 第 7 节：模型组件消融实验
- 7.1 实验目的
- 7.2 消融变体设计（6 种变体表格）
- 7.3 实验设置
- 7.4 消融实验结果（待填写表格）
- 7.5 消融分析（示例要点）
- 7.6 复现命令

### 第 8 节：泛化性实验（跨项目预测）
- 8.1 实验目的
- 8.2 实验场景设计（3 种场景表格）
- 8.3 数据切分策略
- 8.4 项目选择与特征对齐
- 8.5 泛化实验结果（待填写表格）
- 8.6 泛化性分析（示例要点）
- 8.7 复现命令

### 第 9 节：假设检验
- 9.1 检验目的
- 9.2 假设检验方法论
  - 9.2.1 Friedman Test
  - 9.2.2 Nemenyi Post-hoc Test
  - 9.2.3 替代方案（Kruskal-Wallis + Dunn）
- 9.3 实验设置
- 9.4 假设检验结果（待填写）
  - 9.4.1 回归任务
  - 9.4.2 分类任务
- 9.5 统计显著性分析（示例要点）
- 9.6 可视化结果
- 9.7 复现命令
- 9.8 参考文献（Demšar 2006）

### 第 10 节：综合结论与建议（level2 补充）
- 10.1 消融实验结论
- 10.2 泛化性结论
- 10.3 假设检验结论
- 10.4 实践建议

### 附录：实验代码结构

---

## 🎯 实验设计亮点

### 1. 消融实验设计合理
- 覆盖关键组件：共享层、专家机制、塔结构、网络深度、正则化
- 完整模型作为基线，逐步移除/简化
- 保持训练配置一致，确保公平比较

### 2. 泛化性实验贴合真实场景
- Within-Project 作为性能上界
- Cross-Project 评估零样本迁移
- Mixed-Training 评估多源学习效果
- 提供特征对齐策略（词表统一、未知类别处理）

### 3. 假设检验方法严谨
- 基于 Demšar (2006) 推荐的非参数方法
- Friedman test 作为主检验（稳健、适合小样本）
- Nemenyi post-hoc 控制 Type I 错误
- 提供替代方案（Kruskal-Wallis + Dunn）
- 强调效应大小（平均秩次差）而非仅看 p 值

### 4. 文档完善
- 三层文档：快速入门 → 详细说明 → 实验报告
- 提供完整的复现命令
- 包含方法论参考文献
- 示例分析要点（待结果后补充）

---

## 📝 使用流程

### 步骤 1：运行基础实验（level1 & level3）

```powershell
python main.py --config configs\django_shared.yaml
python main.py --config configs\django_mmoe.yaml
```

### 步骤 2：运行消融实验

```powershell
python experiments\ablation_study.py --config configs\ablation_mmoe.yaml --output outputs\ablation
```

### 步骤 3：运行泛化性实验

```powershell
python experiments\cross_project.py --config configs\django_shared.yaml --output outputs\cross_project --projects django
```

### 步骤 4：运行假设检验

```powershell
python experiments\hypothesis_testing.py --summaries outputs\ablation\ablation_summary.json --output outputs\hypothesis_testing --metrics R2 RMSE MAE Accuracy F1 F1_macro
```

### 步骤 5：生成可视化

```powershell
python experiments\visualize_experiments.py --ablation outputs\ablation\ablation_summary.json --cross_project outputs\cross_project\generalization_summary.json --output outputs\visualizations
```

### 步骤 6：填写实验报告

1. 打开 `REPORT.md`
2. 查看各实验输出的 JSON 文件
3. 将结果填入对应表格
4. 根据结果撰写分析文字

---

## 🔧 技术实现细节

### 消融实验核心逻辑

```python
# 定义消融变体
VARIANTS = {
    "full": {"modifications": {}},
    "no_shared": {"modifications": {"bottom_mlp": []}},
    "shallow": {"modifications": {"bottom_mlp": [64], ...}},
    # ...
}

# 应用修改并构建模型
def build_ablation_model(base_cfg, variant):
    model_cfg = base_cfg.copy()
    model_cfg.update(VARIANTS[variant]["modifications"])
    return MTLModel(**model_cfg)
```

### 假设检验核心逻辑

```python
# Friedman test
def friedman_test(data):  # data: (n_datasets, n_algorithms)
    return friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])

# Nemenyi test
def nemenyi_test(data, alpha=0.05):
    ranks = compute_ranks(data)  # 对每个数据集排秩
    mean_ranks = ranks.mean(axis=0)
    cd = q_alpha * sqrt(k*(k+1)/(6*N))  # 临界差值
    return pairwise_compare(mean_ranks, cd)
```

### 跨项目数据对齐

```python
# 构建统一词表
all_categories = set()
for project in projects:
    df = load_project_data(project)
    all_categories.update(df[cat_col].unique())

# 映射未知类别
df[cat_col] = df[cat_col].map(lambda x: x if x in vocab else "<UNK>")
```

---

## ⚠️ 注意事项

### 1. 数据要求

- **消融实验**：使用单一项目数据即可
- **跨项目实验**：需要多个项目的数据文件（放在 `data/engineered/` 目录）
- **假设检验**：需要先运行消融或泛化实验获得结果

### 2. 计算资源

- 消融实验会训练多个模型变体，建议在 GPU 上运行
- 可通过减少 epochs 或只运行部分变体来加速

### 3. 统计显著性

- Friedman test 要求至少 3 个算法和 5 个数据集
- 单数据集场景下，可使用多次随机切分或 k-fold 交叉验证
- 或使用配对 t 检验 / Wilcoxon signed-rank test（需修改代码）

### 4. 结果填写

- 运行实验后，查看输出的 JSON 文件
- 复制对应指标值到 `REPORT.md` 的表格中
- **不要只有数字**，需根据结果撰写分析文字（参考报告中的"待结果后补充"提示）

---

## 🎓 学习资源

### 消融实验
- Goodfellow et al. (2016). *Deep Learning*. Chapter 7: Regularization

### 泛化性与迁移学习
- Pan & Yang (2010). A Survey on Transfer Learning. *IEEE TKDE*, 22(10), 1345-1359.

### 假设检验
- **Demšar, J. (2006)**. Statistical comparisons of classifiers over multiple data sets. *JMLR*, 7, 1-30. ⭐
- García & Herrera (2008). An Extension on "Statistical Comparisons of Classifiers over Multiple Data Sets". *JMLR*, 9, 2677-2694.

---

## ✨ 扩展建议

如果时间充裕，可以进一步实现：

### 1. 更多消融变体
- 不同损失权重组合（`w_reg` vs `w_cls`）
- 特征消融（移除某类特征）
- 不同激活函数（ReLU vs LeakyReLU vs Tanh）

### 2. 更复杂的泛化场景
- 领域自适应（Domain Adaptation）：对抗训练 + 领域分类器
- 迁移学习（Transfer Learning）：预训练 + 微调
- 元学习（Meta Learning）：MAML 等

### 3. 更多统计方法
- Bootstrap 置信区间
- Bayesian 统计推断（如 Bayesian signed-rank test）
- 效应大小分析（Cohen's d）

### 4. 可解释性分析
- 特征重要性（SHAP 值、permutation importance）
- 注意力权重可视化（针对 MMoE 的 gate）
- 决策边界可视化（t-SNE 降维）

---

## 📞 FAQ

### Q: 如何验证实验代码是否正确？

A: 运行单元测试：
```powershell
python experiments\test_experiments.py
```

### Q: 消融实验结果差异很小怎么办？

A: 可能原因：
1. 数据集较简单，模型容量过剩
2. 随机性导致的波动
3. 需要更多数据集或更多次运行

建议：使用假设检验判断是否显著。

### Q: 没有多项目数据如何做泛化实验？

A: 可以：
1. 使用单项目的不同时间段作为"伪项目"
2. 使用 k-fold 交叉验证
3. 或在报告中说明"待补充多项目数据"

### Q: Friedman test 不显著怎么办？

A: 正常现象，说明：
1. 算法间差异不大（都不错或都不好）
2. 数据集数量不足，检验力较弱

建议：报告实际数值 + 说明"无显著差异"，选择更简单/高效的模型。

---

## ✅ 检查清单

提交前确认：

- [ ] 代码可运行（至少运行过一次完整流程）
- [ ] 实验结果已填入 `REPORT.md`
- [ ] 分析文字已撰写（不能只有表格）
- [ ] 图表已生成并嵌入报告
- [ ] 复现命令已测试
- [ ] 文档无错别字

---

## 🎉 完成标志

当你看到以下输出时，说明 level2 实验已完成：

```
outputs/
├── ablation/
│   └── ablation_summary.json ✅
├── cross_project/
│   └── generalization_summary.json ✅
├── hypothesis_testing/
│   └── hypothesis_testing_summary.json ✅
└── visualizations/
    ├── ablation_comparison.png ✅
    ├── cross_project_comparison.png ✅
    └── comprehensive_results.md ✅
```

并且 `REPORT.md` 第 6-10 节已填写完整！

---

**祝实验顺利！如有问题，请查阅 `experiments/README.md` 或 `experiments/QUICKSTART.md`。**
