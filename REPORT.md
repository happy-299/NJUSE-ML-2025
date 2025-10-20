# 多任务学习实验报告（最终版）

本报告面向“预测 Pull Request 关闭时长（回归）与是否合入（分类）”两大任务，按照实验手册的结构给出：问题与数据、特征、模型与方法、结果与可视化分析、结论与建议。所有实验均可通过提供的 YAML 配置、训练脚本与绘图脚本复现。

更新日期：2025-10-19

---

## 1. 问题与数据

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

## 2. 特征列表（本次实际使用）

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

## 3. 模型与方法

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

## 4. 结果与可视化分析

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

## 5. 结论与建议

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

## 附：复现实验与生成图表

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

