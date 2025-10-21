# 多任务学习实验报告

本报告面向“预测 Pull Request 关闭时长（回归）与是否合入（分类）”两大任务，按照实验手册的结构给出：问题与数据、特征、模型与方法、结果与可视化分析、结论与建议。所有实验均可通过提供的 YAML 配置、训练脚本与绘图### 附：复现实验与生成图表

- 训练 Shared-Bottom：
  ```powershell
  python -m src.mtl.train --config configs\django_shared.yaml
  ```
- 训练 MMoE：
  ```powershell
  python -m src.mtl.train --config configs\django_mmoe.yaml
  ```
- 生成图表（基于已有 outputs 目录）：
  ```powershell
  python scripts\plot_results.py .\outputs\django_shared .\outputs\django_mmoe --out .\outputs\figures
  ```
- 导出训练使用的特征（用于核对"特征列表"）：
  ```powershell
  python scripts\dump_meta.py .\outputs\django_shared
  ```

---

## level2

### 6. 特征消融实验


**特征消融实验**
将所有应用的特征分为多个类别，每次实验禁用一个类别的特征训练模型，得到结果如下：

**回归任务性能对比**：

|                 |       R2 |    RMSE |     MAE |         MSE |
| :-------------- | -------: | ------: | ------: | ----------: |
| 所有特征        | 0.553455 | 2450.94 |   709.7 | 6.00713e+06 |
| 禁用-PR基本信息 | 0.606438 | 2300.95 |  642.82 | 5.29438e+06 |
| 禁用-代码变更   | 0.553455 | 2450.94 |   709.7 | 6.00713e+06 |
| 禁用-文本语义   |  0.61838 | 2265.77 | 697.572 | 5.13373e+06 |
| 禁用-评审协作   | 0.553455 | 2450.94 |   709.7 | 6.00713e+06 |
| 禁用-作者经验   | 0.586936 | 2357.27 | 725.625 | 5.55673e+06 |
| 禁用-网络中心性 | 0.586532 | 2358.42 | 728.226 | 5.56216e+06 |
| 禁用-项目上下文 | 0.553455 | 2450.94 |   709.7 | 6.00713e+06 |
| 禁用-时间模式   | 0.566259 | 2415.55 | 720.516 | 5.83489e+06 |
| 禁用-语义标签   | 0.553455 | 2450.94 |   709.7 | 6.00713e+06 |
| 禁用-复杂度指标 | 0.553455 | 2450.94 |   709.7 | 6.00713e+06 |

![回归结果](outputs\figures\django_real_mmoe_reg_ablation.png)


**分类任务性能对比**：

|                 | Accuracy | Precision |   Recall |       F1 | F1_macro |
| :-------------- | -------: | --------: | -------: | -------: | -------: |
| 所有特征        | 0.828426 |  0.873134 | 0.756669 | 0.810741 | 0.826914 |
| 禁用-PR基本信息 | 0.757754 |  0.768166 | 0.717866 | 0.742165 | 0.756865 |
| 禁用-代码变更   | 0.828426 |  0.873134 | 0.756669 | 0.810741 | 0.826914 |
| 禁用-文本语义   |  0.69415 |  0.651255 |  0.79709 |  0.71683 | 0.692175 |
| 禁用-评审协作   | 0.828426 |  0.873134 | 0.756669 | 0.810741 | 0.826914 |
| 禁用-作者经验   | 0.783274 |  0.731889 | 0.873888 |  0.79661 | 0.782339 |
| 禁用-网络中心性 | 0.886141 |  0.940465 |   0.8173 | 0.874567 | 0.885163 |
| 禁用-项目上下文 | 0.828426 |  0.873134 | 0.756669 | 0.810741 | 0.826914 |
| 禁用-时间模式   | 0.888496 |   0.90279 | 0.863379 | 0.882645 | 0.888218 |
| 禁用-语义标签   | 0.828426 |  0.873134 | 0.756669 | 0.810741 | 0.826914 |
| 禁用-复杂度指标 | 0.828426 |  0.873134 | 0.756669 | 0.810741 | 0.826914 |

![分类结果](outputs\figures\django_real_mmoe_feature_ablation_by_group.png)



### 7. 模型组件消融实验

#### 7.1 实验目的

评估多任务学习模型中各关键组件对最终性能的贡献度，识别哪些组件是必要的，哪些可以简化。

#### 7.2 消融变体设计

基于完整模型（Full Model），设计以下消融变体：

| 变体名称               | 描述             | 修改内容                                  |
| ---------------------- | ---------------- | ----------------------------------------- |
| **Full**         | 完整模型（基线） | 完整的 Shared-Bottom 或 MMoE 结构         |
| **No Shared**    | 无共享底层       | 移除共享底层 MLP，每个任务独立建模        |
| **Single Tower** | 单塔结构         | 回归和分类使用相同的塔结构                |
| **No Expert**    | 单专家 MMoE      | MMoE 退化为只有 1 个专家（类似共享底层）  |
| **Shallow**      | 浅层网络         | 减少所有隐藏层深度（如 [128,64] → [64]） |
| **No Dropout**   | 无正则化         | 移除所有 Dropout 层                       |

#### 7.3 实验设置

- **数据集**：Django PR 数据（与 level1 一致）
- **评估指标**：
  - 回归：R2、RMSE、MAE、MSE
  - 分类：Accuracy、Percision、Recall、F1、F1_macro
- **训练配置**：与完整模型保持一致（batch_size=256, lr=1e-3, epochs=50, early_stopping=5）
- **随机种子**：42（确保可复现）

#### 7.4 消融实验结果


**回归任务性能对比**：



**分类任务性能对比**：




#### 7.5 消融分析

（待结果后补充，示例分析要点）：

1. **共享底层的作用**：

   - 若 No Shared 性能显著下降 → 共享表示对多任务学习至关重要
   - 若性能无明显变化 → 两任务相关性较弱，可能需要更独立的建模
2. **专家机制的贡献**（针对 MMoE）：

   - 若 No Expert 性能接近 Full → 多专家机制未充分利用
   - 若 No Expert 显著下降 → 多专家能有效捕获任务异质性
3. **网络深度影响**：

   - Shallow 性能如何权衡效率与表达能力
4. **正则化效果**：

   - No Dropout 是否过拟合

#### 7.6 复现命令

```powershell
# 运行完整消融实验（所有变体）
python experiments\ablation_study.py --config configs\django_mmoe.yaml --output outputs\ablation

# 运行特定变体
python experiments\ablation_study.py --config configs\django_mmoe.yaml --output outputs\ablation --variants full no_shared shallow

# 可视化消融结果
python experiments\visualize_experiments.py --ablation outputs\ablation\ablation_summary.json --output outputs\visualizations
```

---

### 8. 泛化性实验：跨项目预测

#### 8.1 实验目的

评估模型在不同项目间的迁移能力，回答以下问题：

1. 在项目 A 训练的模型能否泛化到项目 B？
2. 混合多项目训练是否提升单项目性能？
3. 哪些特征/模式在跨项目场景下更稳健？

#### 8.2 实验场景设计

| 场景                         | 训练集             | 测试集        | 目的         |
| ---------------------------- | ------------------ | ------------- | ------------ |
| **Within-Project**     | Django (80%)       | Django (20%)  | 项目内基线   |
| **Cross-Project-A→B** | Django (100%)      | Flask (100%)  | 单向迁移     |
| **Cross-Project-B→A** | Flask (100%)       | Django (100%) | 反向迁移     |
| **Mixed-Training**     | Django+Flask (80%) | Django (20%)  | 混合训练评估 |

**说明**：

- Within-Project 作为性能上界（基线）
- Cross-Project 评估零样本迁移能力
- Mixed-Training 评估多源学习效果

#### 8.3 数据切分策略

为防止数据泄漏并反映真实场景：

1. **项目内（Within-Project）**：

   - 按时间前向切分（推荐）：训练集用早期 PR，测试集用晚期 PR
   - 或随机分层切分（当前实现）
2. **跨项目（Cross-Project）**：

   - 训练项目：使用全部数据（或按时间切分留出验证集）
   - 测试项目：使用全部数据作为测试集，模拟零样本迁移
   - 注意：需确保特征分布一致性（如类别特征词表统一处理）
3. **混合训练（Mixed-Training）**：

   - 训练集：合并多项目的训练部分
   - 验证集：混合或目标项目的验证集
   - 测试集：目标项目的测试集

#### 8.4 项目选择与特征对齐

**可用项目**（示例）：

- Django：大型 Web 框架，PR 数量多，变更复杂度高
- Flask：轻量级 Web 框架，PR 数量中等
- Requests：HTTP 库，PR 变更相对简单

**特征对齐策略**：

1. **数值特征**：使用训练集的 StandardScaler 拟合，应用于所有项目
2. **类别特征**：
   - 构建统一词表（包含所有项目的类别值）
   - 对未见类别映射到 `<UNK>` 索引
3. **项目特定特征**（可选）：
   - 添加 `project_id` 作为类别特征
   - 或使用 domain-adaptation 技术（如对抗训练）

#### 8.5 泛化实验结果

（待实验运行后填写）

**回归任务（R2 Score）**：

| 场景           | 训练集       | 测试集 | R2    | RMSE    | 相对基线 |
| -------------- | ------------ | ------ | ----- | ------- | -------- |
| Within-Project | Django       | Django | 0.619 | 2316.31 | baseline |
| Cross-Project  | Django       | Flask  | -     | -       | -        |
| Cross-Project  | Flask        | Django | -     | -       | -        |
| Mixed-Training | Django+Flask | Django | -     | -       | -        |

**分类任务（Accuracy & F1）**：

| 场景           | 训练集       | 测试集 | Accuracy | F1    | F1_macro | 相对基线 |
| -------------- | ------------ | ------ | -------- | ----- | -------- | -------- |
| Within-Project | Django       | Django | 0.696    | 0.588 | 0.674    | baseline |
| Cross-Project  | Django       | Flask  | -        | -     | -        | -        |
| Cross-Project  | Flask        | Django | -        | -     | -        | -        |
| Mixed-Training | Django+Flask | Django | -        | -     | -        | -        |

#### 8.6 泛化性分析

（待结果后补充，示例分析要点）：

1. **跨项目性能下降幅度**：

   - 如果下降 > 20% → 项目间差异显著，需 domain-adaptation
   - 如果下降 < 10% → 模型具备良好泛化能力
2. **特征稳健性**：

   - 哪些特征在跨项目场景下权重稳定？
   - 项目特定特征（如 author_experience）是否影响迁移？
3. **混合训练效果**：

   - 是否存在负迁移（negative transfer）？
   - 数据增强是否提升单项目性能？

#### 8.7 复现命令

```powershell
# 项目内基线（Django）
python experiments\cross_project.py --config configs\django_shared.yaml --output outputs\cross_project --projects django

# 跨项目实验（需要多项目数据）
python experiments\cross_project.py --config configs\django_shared.yaml --output outputs\cross_project --projects django flask

# 可视化泛化结果
python experiments\visualize_experiments.py --cross_project outputs\cross_project\generalization_summary.json --output outputs\visualizations
```

---

### 9. 假设检验：评估算法显著性差异

#### 9.1 检验目的

在比较多个算法（如 Shared-Bottom vs MMoE vs 消融变体）时，需要统计检验评估：

1. 算法间是否存在**显著差异**（overall test）
2. 哪些算法对之间**显著不同**（post-hoc pairwise comparison）

#### 9.2 假设检验方法论

基于 **Demšar (2006)** 的推荐方法：

##### 9.2.1 Friedman Test（主检验）

- **适用场景**：多个算法在多个数据集上的非参数比较
- **原假设 H₀**：所有算法性能无显著差异（秩次相同）
- **优点**：
  - 不假设数据分布（非参数）
  - 适合小样本量
  - 对异常值稳健

**检验流程**：

1. 对每个数据集，对 k 个算法的性能排秩（rank）
2. 计算 Friedman 统计量：
   ```
   χ²_F = (12N / k(k+1)) * Σ(R_j² - k(k+1)²/4)
   ```

   其中 N 是数据集数，k 是算法数，R_j 是算法 j 的平均秩次
3. 若 p < α（如 0.05），拒绝 H₀ → 进行 post-hoc 检验

##### 9.2.2 Nemenyi Post-hoc Test

- **触发条件**：Friedman test 显著（p < α）
- **目的**：成对比较，识别具体哪些算法对显著不同
- **临界差值（Critical Difference, CD）**：
  ```
  CD = q_α * sqrt(k(k+1) / 6N)
  ```

  其中 q_α 是 studentized range statistic
- **判定规则**：若两算法平均秩次差 > CD，则显著不同

**注意**：Nemenyi 相对保守（类似 Tukey HSD），适合对所有算法对进行比较。

##### 9.2.3 替代方案：Kruskal-Wallis + Dunn's Test

- **Kruskal-Wallis**：类似 Friedman，但假设较弱
- **Dunn's Test**：post-hoc 方法，可配合 Bonferroni 校正

**选择建议**：

- 默认使用 **Friedman + Nemenyi**（Demšar 推荐）
- 若数据集数较少（< 5），可改用 **配对 t 检验** 或 **Wilcoxon signed-rank test**

#### 9.3 实验设置

- **算法**：Shared-Bottom, MMoE, 及各消融变体（共 6-8 个）
- **数据集**：
  - 理想情况：多个项目数据（Django, Flask, Requests, ...）
  - 当前情况：单项目多次随机切分（或 k-fold 交叉验证）
- **指标**：R2（回归）、Accuracy（分类）、F1_macro（分类）
- **显著性水平**：α = 0.05

#### 9.4 假设检验结果

（待实验运行后填写）

##### 9.4.1 回归任务（R2）

**Friedman Test**：

- 统计量 χ²_F = -
- p-value = -
- 结论：（拒绝/不拒绝）原假设

**Nemenyi Post-hoc Test**（如果 Friedman 显著）：

- 临界差值 CD = -
- 平均秩次：
  - Full Model: -
  - MMoE: -
  - Shared-Bottom: -
  - No Shared: -
  - Shallow: -
  - ...

**成对比较矩阵**（显著差异标记 *）：

|                      | Full | MMoE | Shared | No Shared | Shallow | No Dropout |
| -------------------- | ---- | ---- | ------ | --------- | ------- | ---------- |
| **Full**       | -    |      |        |           |         |            |
| **MMoE**       |      | -    |        |           |         |            |
| **Shared**     |      |      | -      |           |         |            |
| **No Shared**  |      |      |        | -         |         |            |
| **Shallow**    |      |      |        |           | -       |            |
| **No Dropout** |      |      |        |           |         | -          |

（\* 表示 p < 0.05，\*\* 表示 p < 0.01，\*\*\* 表示 p < 0.001）

##### 9.4.2 分类任务（Accuracy）

（类似结构）

#### 9.5 统计显著性分析

（待结果后补充，示例要点）：

1. **主检验结论**：

   - 若 Friedman 显著 → "算法间存在显著差异，需进一步 post-hoc 分析"
   - 若不显著 → "各算法性能无显著差异，可能因数据集单一或样本量不足"
2. **成对比较解读**：

   - 示例："MMoE 在回归任务上显著优于 No Shared (p=0.003)，但与 Shared-Bottom 无显著差异 (p=0.42)"
   - 关注**效应大小**（平均秩次差）而非仅看 p 值
3. **实践建议**：

   - 若算法 A 与 B 无显著差异，优先选择更简单/高效的模型
   - 若某消融变体显著下降，说明该组件关键
4. **局限性**：

   - 单数据集场景下，Friedman test 检验力较弱
   - 建议扩展到多项目或使用 bootstrap 置信区间

#### 9.6 可视化结果

1. **算法性能箱线图**：展示各算法在多个数据集上的分布
2. **Nemenyi 临界差值图**：平均秩次 + CD 标记
3. **成对比较热图**：p 值矩阵，颜色深浅表示显著性

#### 9.7 复现命令

```powershell
# 假设检验（基于消融实验结果）
python experiments\hypothesis_testing.py --summaries outputs\ablation\ablation_summary.json --output outputs\hypothesis_testing --metrics R2 RMSE MAE Accuracy F1 F1_macro --test friedman

# 可选：使用 Kruskal-Wallis
python experiments\hypothesis_testing.py --summaries outputs\ablation\ablation_summary.json --output outputs\hypothesis_testing --test kruskal

# 查看可视化结果
# 输出：outputs\hypothesis_testing\hypothesis_test_R2.png
# 输出：outputs\hypothesis_testing\hypothesis_test_R2_heatmap.png
```

#### 9.8 参考文献

> Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, 7, 1-30.

关键要点（原文）：

- **推荐 Friedman test** 作为主检验（非参数、稳健）
- **Nemenyi test** 用于 post-hoc 全比较（保守但安全）
- **不推荐多次 t 检验**（会增加 Type I 错误率）
- **报告平均秩次和 CD**，而非仅报告 p 值
- **关注效应大小**，避免"显著但无实际意义"的结论

---

### 10. 综合结论与建议（level2 补充）

#### 10.1 消融实验结论

（待结果）：

- 共享底层 / 多专家机制的关键性
- 网络深度与性能的权衡
- 正则化的有效性

#### 10.2 泛化性结论

（待结果）：

- 模型在跨项目场景下的迁移能力
- 项目间差异的主要来源
- 混合训练的利弊

#### 10.3 假设检验结论

（待结果）：

- 哪些算法间存在统计显著差异
- 最优算法的选择依据（兼顾显著性和实用性）

#### 10.4 实践建议

1. **模型选择**：

   - 若分类召回优先 → Shared-Bottom
   - 若回归精度优先 → MMoE
   - 若计算资源受限 → 浅层变体
2. **特征工程**：

   - 跨项目迁移时，优先使用通用特征（如 PR 规模、评审活跃度）
   - 避免过度依赖项目特定特征（如特定 author ID）
3. **训练策略**：

   - 单项目场景：时间前向切分 + 早停
   - 跨项目场景：domain-adaptation 或混合训练 + 项目 ID 特征
4. **统计评估**：

   - 始终进行假设检验，避免"偶然性优势"
   - 报告置信区间或标准差，体现结果稳定性

---

### 附录：实验代码结构

```
experiments/
├── ablation_study.py          # 消融实验
├── cross_project.py           # 泛化性实验
├── hypothesis_testing.py      # 假设检验
├── run_all.py                 # 一键运行所有实验
├── visualize_experiments.py   # 结果可视化
└── README.md                  # 实验说明文档
```

详细使用说明见 `experiments/README.md`。-10-19

---

## level1 & level3

### 1. 问题与数据

- 任务定义：

  - 任务一（回归）：预测 PR 关闭所需时间（目标字段：`processing_time`），主要指标：MAE、MSE、RMSE、R2。
  - 任务二（分类）：预测 PR 是否被合并（目标字段：`merged`，二分类），主要指标：Accuracy、Precision、Recall、F1（含 Macro）。
- 仓库数量与数据来源：

  - 本轮实验使用 `engineered/django_engineered.xlsx`（工程化后的 Django 仓库 PR 特征表）。如扩展至多仓库，可在数据中加入仓库 ID/组织信息并纳入特征。
- 数据切分与防止泄漏：

  - 当前采用随机切分并对 `merged` 分层：test_size=0.2、val_size=0.1（配置见 `configs/django_*.yaml`）。
  - 防泄漏措施：
    1) 仅使用 PR 创建时即可获得的特征作为输入；
    2) 目标列仅用于构造标签；
    3) 标准化器在训练集拟合并在验证/测试复用；
    4) 若数据包含时间戳（如 created_at/closed_at），建议改用时间前向切分（训练用过去，验证/测试用将来），避免时间泄漏。

---

### 2. 特征列表（本次实际使用）

类别特征（12）：

```
state, has_test, has_feature, has_bug, has_document, has_improve, has_refactor,
is_reviewed, last_comment_mention, has_bug_keyword, has_feature_keyword, has_document_keyword
```

数值特征（从 checkpoint meta 导出，示例共 110 列，节选）：

```
number, comments, review_comments, commits, additions, deletions, changed_files,
conversation, directory_num, language_num, file_type, has_test, has_feature, has_bug,
has_document, has_improve, has_refactor, title_length, title_readability, body_length,
body_readability, lines_added, lines_deleted, segs_added, segs_deleted, segs_updated,
files_added, files_deleted, files_updated, modify_proportion, modify_entropy,
test_churn, non_test_churn, reviewer_num, bot_reviewer_num, is_reviewed, comment_num,
comment_length, last_comment_mention, experience, is_reviewer, change_num, participation,
changes_per_week, avg_round, avg_duration, merge_proportion, degree_centrality,
closeness_centrality, betweenness_centrality, eigenvector_centrality, k_coreness,
experience_reviewer, is_author, change_num_reviewer, participation_reviewer, avg_comments,
avg_files, avg_round_reviewer, avg_duration_reviewer, merge_proportion_reviewer,
degree_centrality_reviewer, closeness_centrality_reviewer, betweenness_centrality_reviewer,
eigenvector_centrality_reviewer, k_coreness_reviewer, project_age, open_changes, author_num,
team_size, changes_per_author, changes_per_reviewer, avg_lines, avg_segs, add_per_week,
del_per_week, avg_reviewers, avg_rounds, avg_rounds_merged, avg_duration_merged,
avg_churn_merged, avg_files_merged, avg_comments_merged, avg_rounds_abandoned,
avg_duration_abandoned, avg_churn_abandoned, avg_files_abandoned, avg_comments_abandoned,
comment_count, avg_comment_length, commit_count, total_additions, total_deletions,
total_files_changed, last_response_time, created_hour, created_dayofweek, created_month,
has_bug_keyword, has_feature_keyword, has_document_keyword, total_changes, net_changes,
change_density, additions_per_file, author_experience, author_activity, reviewer_count,
author_exp_change_size, complexity_per_reviewer
```

说明：

- 数值特征由数据表中的数值列自动推断（排除 `processing_time`、`merged`）。
- 类别特征使用 embedding 方式（one_hot_categorical=false）。
- 具体列表可通过以下命令导出（与当前 run 保持一致）：
  ```powershell
  python scripts\dump_meta.py .\outputs\django_shared
  ```

---

### 3. 模型与方法

- 多任务学习（MTL）结构：

  - Shared-Bottom：共享底部 MLP，接回归塔与分类塔。
  - MMoE：多专家 + 任务门控，为不同任务自适应选择专家子空间（本报告含对比）。
- 输入编码：

  - 数值：StandardScaler 标准化；
  - 类别：Embedding（embedding_dim=8，词表含未知索引0）。
- 联合损失：

  - Regression：MSE；Classification：BCE（sigmoid 概率）；
  - 组合：loss = w_reg*MSE + w_cls*BCE（本次 w_reg=w_cls=1）。
- 训练策略：

  - 优化器 Adam，早停监控 `val_loss`，保存 `best.pt`；
  - 批大小 256、学习率 1e-3，训练 8 轮（可调），分层采样保证类别分布稳定。

---

### 4. 结果与可视化分析

图表（已生成，见 `outputs/figures/`）：

- `loss.png`：训练/验证损失曲线；
- `reg_R2.png`：回归 R2 对比（越高越好）；
- `reg_errors.png`：回归 RMSE/MAE（越低越好）；
- `cls_scores.png`：分类 Accuracy/F1/F1_macro 对比（越高越好）。

测试集量化结果：

- Shared-Bottom（`outputs/django_shared`）：
  - 回归：R2≈0.609，RMSE≈2348.65，MAE≈718.98
  - 分类：Accuracy≈0.700，Precision≈0.727，Recall≈0.612，F1≈0.665，F1_macro≈0.697
- MMoE（`outputs/django_mmoe`）：
  - 回归：R2≈0.619，RMSE≈2316.31，MAE≈707.80
  - 分类：Accuracy≈0.696，Precision≈0.862，Recall≈0.446，F1≈0.588，F1_macro≈0.674

分析要点：

- 两者都稳定收敛；MMoE 在回归上略有优势（误差更低、R2 更高），但 Shared-Bottom 的分类 Recall/F1 更高，体现了共享表示对分类召回的帮助。
- 若分类召回是业务优先目标（减少漏判可合入 PR），Shared-Bottom 更合适；若回归误差要求更严苛，MMoE 更具优势。
- 进一步优化方向：对 MMoE 增加专家数/加深塔层/调节损失权重，或引入更细粒度的类别/文本特征（如变更文件类型嵌入、评论语义 embedding 等）。

---

### 5. 结论与建议

- 结论：

  1) 多任务学习在本数据上取得兼顾两任务的稳定表现；
  2) Shared-Bottom 更有利于分类召回与 F1，MMoE 在回归拟合上略优；
  3) 通过增加特征维度、调整网络结构/损失权重与时间切分评估，可进一步提升泛化与稳健性。
- 对维护者与评审流程的建议：

  1) 在 PR 创建阶段补充可用结构化信号（改动规模、文件类型分布、作者/评审者活跃度、关键字粒度计数等），有助于早期预估合入概率与处理时长；
  2) 将 `processing_time` 预测用于评审排期，对预计耗时较长的 PR 提前匹配有经验的审阅者；
  3) 采用时间前向切分建立长期评估基线，周期性重训与校准分类阈值以匹配业务偏好（如更高 Recall）；
  4) 扩展到多仓库数据时加入仓库/组织特征，并尝试 MMoE 或 domain-adaptation 以提升跨仓库泛化。

---

### 附：复现实验与生成图表

- 训练 Shared-Bottom：
  ```powershell
  python -m src.mtl.train --config configs\django_shared.yaml
  ```
- 训练 MMoE：
  ```powershell
  python -m src.mtl.train --config configs\django_mmoe.yaml
  ```
- 生成图表（基于已有 outputs 目录）：
  ```powershell
  python scripts\plot_results.py .\outputs\django_shared .\outputs\django_mmoe --out .\outputs\figures
  ```
- 导出训练使用的特征（用于核对“特征列表”）：
  ```powershell
  python scripts\dump_meta.py .\outputs\django_shared
  ```
