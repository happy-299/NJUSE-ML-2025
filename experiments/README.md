# 实验模块

本目录包含各类扩展实验的实现。

## 文件说明

- `ablation_study.py`: 模型组件消融实验
- `cross_project.py`: 跨项目泛化性实验
- `hypothesis_testing.py`: 假设检验（Friedman test、Nemenyi test 等）
- `run_all.py`: 统一运行所有实验的脚本
- `visualize_experiments.py`: 实验结果可视化

## 使用方法

### 1. 消融实验

```powershell
python experiments\ablation_study.py --config configs\django_shared.yaml --output outputs\ablation
```

消融实验变体：
- `full`: 完整模型
- `no_shared`: 无共享底层
- `single_tower`: 单塔结构
- `no_expert`: 单专家 MMoE
- `shallow`: 浅层网络
- `no_dropout`: 无 Dropout

### 2. 泛化性实验

```powershell
python experiments\cross_project.py --config configs\django_shared.yaml --output outputs\cross_project --projects django
```

### 3. 假设检验

```powershell
python experiments\hypothesis_testing.py --summaries outputs\ablation\ablation_summary.json --output outputs\hypothesis_testing --metrics R2 RMSE MAE Accuracy F1 F1_macro
```

支持的检验方法：
- Friedman test (默认)
- Kruskal-Wallis test
- Nemenyi post-hoc test
- Dunn's post-hoc test

### 4. 一键运行所有实验

```powershell
python experiments\run_all.py --config configs\django_shared.yaml --output outputs
```

### 5. 可视化结果

```powershell
python experiments\visualize_experiments.py --ablation outputs\ablation\ablation_summary.json --cross_project outputs\cross_project\generalization_summary.json --output outputs\visualizations
```

## 实验设计

### 消融实验

目的：评估各模型组件的贡献度

方法：
1. 完整模型作为基线
2. 逐步移除/简化关键组件
3. 对比性能变化

关键组件：
- 共享底层（Shared Bottom）
- 专家机制（MMoE Experts）
- 任务特定塔（Task Towers）
- 正则化（Dropout）
- 网络深度

### 泛化性实验

目的：评估模型的跨项目迁移能力

场景：
1. 项目内评估（基线）：在同一项目的训练/测试集上评估
2. 跨项目迁移：在项目 A 训练，在项目 B 测试
3. 混合训练：在多个项目的混合数据上训练

### 假设检验

目的：评估不同算法间的统计显著性差异

方法（基于 Demšar 2006）：
1. **Friedman test**：非参数检验，检测多个算法在多个数据集上是否有显著差异
2. **Nemenyi post-hoc test**：如果 Friedman test 显著，进行成对比较
3. **Kruskal-Wallis + Dunn's test**：替代方案

参考文献：
> Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1-30.

## 输出结构

```
outputs/
├── ablation/
│   ├── ablation_full/
│   │   ├── best.pt
│   │   ├── history.json
│   │   └── results.json
│   ├── ablation_no_shared/
│   │   └── ...
│   └── ablation_summary.json
├── cross_project/
│   ├── within_django/
│   │   └── ...
│   ├── cross_django_to_flask/
│   │   └── ...
│   └── generalization_summary.json
├── hypothesis_testing/
│   ├── hypothesis_test_R2.json
│   ├── hypothesis_test_R2.png
│   ├── hypothesis_test_R2_heatmap.png
│   └── hypothesis_testing_summary.json
└── visualizations/
    ├── ablation_comparison.png
    ├── cross_project_comparison.png
    └── comprehensive_results.md
```

## 注意事项

1. **数据要求**：
   - 消融实验使用单一项目数据即可
   - 跨项目实验需要多个项目的数据文件
   - 假设检验需要先运行消融或泛化实验获得结果

2. **计算资源**：
   - 消融实验会训练多个模型变体，需要一定时间
   - 建议在 GPU 上运行以加速训练

3. **统计显著性**：
   - 默认显著性水平 α = 0.05
   - Friedman test 要求至少 3 个算法和 5 个数据集
   - 对于单数据集场景，可使用配对 t 检验或 Wilcoxon signed-rank test

## 扩展建议

1. **更多消融变体**：
   - 不同的损失权重组合
   - 不同的特征子集
   - 不同的网络架构

2. **更多泛化场景**：
   - 领域自适应（Domain Adaptation）
   - 迁移学习（Transfer Learning）
   - 元学习（Meta Learning）

3. **更多统计方法**：
   - Bootstrap 置信区间
   - Bayesian 统计推断
   - 效应大小（Effect Size）分析
