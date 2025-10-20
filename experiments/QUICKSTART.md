# Level2 实验快速入门

本指南帮助你快速运行消融实验、泛化性实验和假设检验。

## 前置条件

确保已完成 level1 & level3 的实验，即已有：
- ✅ 训练好的 Shared-Bottom 和 MMoE 模型
- ✅ `outputs/django_shared/` 和 `outputs/django_mmoe/` 目录
- ✅ Django 项目数据

## 实验运行步骤

### 方式一：一键运行所有实验

```powershell
# 使用 Shared-Bottom 配置作为基础
python experiments\run_all.py --config configs\django_shared.yaml --output outputs

# 或使用 MMoE 配置
python experiments\run_all.py --config configs\django_mmoe.yaml --output outputs
```

### 方式二：分步运行

#### 1. 消融实验

```powershell
# 运行所有消融变体（推荐）
python experiments\ablation_study.py --config configs\django_mmoe.yaml --output outputs\ablation

# 或只运行特定变体（更快）
python experiments\ablation_study.py --config configs\django_mmoe.yaml --output outputs\ablation --variants full no_shared shallow
```

**预计时间**：每个变体约 5-15 分钟（取决于数据大小和硬件）

**输出**：
- `outputs/ablation/ablation_full/` - 完整模型结果
- `outputs/ablation/ablation_no_shared/` - 无共享层变体
- `outputs/ablation/ablation_summary.json` - 汇总结果

#### 2. 泛化性实验

```powershell
# 项目内基线（当前只有 Django 数据）
python experiments\cross_project.py --config configs\django_shared.yaml --output outputs\cross_project --projects django

# 如果有多项目数据（需要准备数据文件）
python experiments\cross_project.py --config configs\django_shared.yaml --output outputs\cross_project --projects django flask requests
```

**数据准备**（如需跨项目）：
1. 将其他项目的数据文件放到 `data/engineered/` 目录
2. 更新 `experiments/cross_project.py` 中的 `AVAILABLE_PROJECTS` 配置

**输出**：
- `outputs/cross_project/within_django/` - 项目内基线
- `outputs/cross_project/generalization_summary.json` - 汇总结果

#### 3. 假设检验

```powershell
# 基于消融实验结果运行假设检验
python experiments\hypothesis_testing.py --summaries outputs\ablation\ablation_summary.json --output outputs\hypothesis_testing --metrics R2 RMSE MAE Accuracy F1 F1_macro

# 使用 Kruskal-Wallis test（替代方案）
python experiments\hypothesis_testing.py --summaries outputs\ablation\ablation_summary.json --output outputs\hypothesis_testing --test kruskal
```

**输出**：
- `outputs/hypothesis_testing/hypothesis_test_R2.json` - R2 检验结果
- `outputs/hypothesis_testing/hypothesis_test_R2.png` - 可视化图表
- `outputs/hypothesis_testing/hypothesis_test_R2_heatmap.png` - 成对比较热图
- `outputs/hypothesis_testing/hypothesis_testing_summary.json` - 总汇总

#### 4. 结果可视化

```powershell
# 生成所有可视化图表
python experiments\visualize_experiments.py --ablation outputs\ablation\ablation_summary.json --cross_project outputs\cross_project\generalization_summary.json --output outputs\visualizations
```

**输出**：
- `outputs/visualizations/ablation_comparison.png` - 消融实验对比图
- `outputs/visualizations/cross_project_comparison.png` - 泛化性对比图
- `outputs/visualizations/comprehensive_results.md` - 综合结果表格（Markdown）
- `outputs/visualizations/comprehensive_results.csv` - 综合结果表格（CSV）

## 结果解读

### 消融实验

查看 `outputs/ablation/ablation_summary.json`，关注：
- **R2 变化**：哪个变体下降最多？→ 该组件最重要
- **F1 变化**：分类任务对哪些组件更敏感？

### 泛化性实验

查看 `outputs/cross_project/generalization_summary.json`，关注：
- **Within vs Cross**：跨项目性能下降多少？
- **迁移方向**：A→B 和 B→A 是否对称？

### 假设检验

查看 `outputs/hypothesis_testing/hypothesis_testing_summary.json`，关注：
- **Friedman p-value**：< 0.05 说明算法间有显著差异
- **Nemenyi 临界差值**：超过 CD 的秩次差表示显著
- **成对比较**：哪些算法对显著不同？

## 常见问题

### Q1: 消融实验太慢怎么办？

**方案**：
- 减少训练轮数（修改配置文件 `epochs: 50` → `epochs: 20`）
- 只运行部分变体（使用 `--variants` 参数）
- 在 GPU 上运行

### Q2: 没有多项目数据如何做泛化实验？

**方案**：
- 使用单项目的不同时间段作为"伪项目"
- 或使用 k-fold 交叉验证模拟多数据集
- 或标注"待补充"，说明实验设计

### Q3: 假设检验要求多个数据集，但只有一个怎么办？

**方案**：
- 使用多次随机切分（改变随机种子）
- 使用 k-fold 交叉验证
- 或使用配对 t 检验 / Wilcoxon signed-rank test（需修改代码）

### Q4: 如何在报告中填写结果？

**步骤**：
1. 运行实验脚本
2. 查看输出的 JSON 文件（如 `ablation_summary.json`）
3. 复制对应指标值到 `REPORT.md` 的表格中
4. 根据数值写分析文字（参考报告中的"待结果后补充"提示）

## 检查清单

在提交报告前，确保：

- [ ] 已运行消融实验（至少 3 个变体）
- [ ] 已运行泛化性实验（至少项目内基线）
- [ ] 已运行假设检验（至少一个指标）
- [ ] 已生成可视化图表
- [ ] 已将结果填入 `REPORT.md` 对应章节
- [ ] 已编写分析文字（不要只有数字表格）
- [ ] 代码可复现（命令可直接运行）

## 时间规划

建议时间分配：
- 消融实验：1-2 小时（含运行和分析）
- 泛化性实验：30 分钟 - 1 小时（单项目）或 2-3 小时（多项目）
- 假设检验：30 分钟
- 可视化与报告撰写：1 小时

总计：3-6 小时

## 进阶实验（可选）

如果时间充裕，可以尝试：

1. **更多消融变体**：
   - 不同损失权重（`w_reg=2, w_cls=1`）
   - 特征消融（移除某类特征）

2. **更复杂的泛化场景**：
   - 领域自适应（Domain Adaptation）
   - 迁移学习（Transfer Learning）

3. **更多统计方法**：
   - Bootstrap 置信区间
   - Bayesian 统计推断
   - 效应大小（Cohen's d）分析

4. **可解释性分析**：
   - 特征重要性（SHAP 值）
   - 注意力权重可视化（针对 MMoE 的 gate）

## 获取帮助

- 查看 `experiments/README.md` 获取详细文档
- 查看各实验脚本的 `--help` 选项
- 检查 `outputs/` 目录下的日志文件

祝实验顺利！🚀
