# Level 2 实验说明

本目录包含所有 level2 要求的实验脚本：
1. **模型组件消融实验** (`ablation_study.py`)
2. **跨项目泛化性实验** (`cross_project.py`)
3. **假设检验评估** (`hypothesis_test.py`)

## 快速开始

### 运行所有实验

```powershell
# 运行所有level2实验（消融、跨项目模型比较、假设检验、可视化）
python experiments\run_all_experiments.py --base_config configs\django_real_mmoe.yaml
```

**实验内容**：
1. **消融实验**：测试模型各组件的重要性（MMoE和Shared-Bottom）
2. **跨项目泛化性演示**：使用3个项目演示跨项目预测能力
3. **跨项目模型比较**：在5个项目上对比MMoE vs Shared-Bottom（用于假设检验）
4. **假设检验**：使用配对统计检验比较两个模型的性能
5. **可视化**：生成所有实验的图表和分析

### 单独运行实验

#### 1. 模型组件消融实验

测试移除或改变模型组件对性能的影响。

```powershell
# 运行MMoE的所有消融实验
python experiments\ablation_study.py --config configs\django_mmoe.yaml --output outputs\ablation_mmoe --ablation all --model_type mmoe

# 运行SharedBottom的所有消融实验
python experiments\ablation_study.py --config configs\django_shared.yaml --output outputs\ablation_shared --ablation all --model_type shared_bottom

# 运行单个消融实验
python experiments\ablation_study.py --config configs\django_mmoe.yaml --output outputs\ablation_mmoe --ablation no_dropout
```

支持的消融类型：
- `no_dropout`: 移除Dropout层
- `shallow_tower`: 使用更浅的任务塔
- `small_embedding`: 使用更小的embedding维度
- `fewer_experts`: 减少专家数量（仅MMoE）
- `single_expert`: 单专家（仅MMoE）
- `all`: 运行所有消融实验

#### 2. 跨项目泛化性实验

测试模型在不同项目间的泛化能力（使用N-1个项目训练，在1个项目测试，Leave-One-Out策略）。

```powershell
# 演示性跨项目实验（3个项目，快速验证）
python experiments\cross_project.py --config configs\django_real_mmoe.yaml --output outputs\cross_project --projects django react tensorflow

# 完整跨项目模型比较（5个项目，用于假设检验）
# MMoE模型
python experiments\cross_project.py --config configs\django_real_mmoe.yaml --output outputs\cross_project_mmoe --projects django react tensorflow moby opencv

# Shared-Bottom模型
python experiments\cross_project.py --config configs\django_real_shared.yaml --output outputs\cross_project_shared --projects django react tensorflow moby opencv
```

**实验说明**：
- 使用Leave-One-Out策略：每次用N-1个项目训练，在剩余1个项目测试
- 项目越多，样本量越大，统计检验越可靠（推荐5+个项目）
- 生成的结果文件包含每个测试项目的详细指标和汇总统计

可用项目（engineered目录下的数据）：
- django, moby, opencv, react, salt
- scikit-learn, symfony, tensorflow, terraform, yii2

#### 3. 假设检验

使用统计方法比较多个模型的性能。根据模型数量自动选择合适的统计检验：
- **两个模型**：Wilcoxon signed-rank test + Paired t-test + Cohen's d
- **三个或更多模型**：Friedman test + Nemenyi post-hoc test

**重要提示：样本数量要求**

为了进行有效的统计检验，每个模型需要**至少2个独立的测量值**（建议5-10个）。获取多个测量值的方法：

1. **多次随机种子训练**：
   ```powershell
   # 使用不同种子训练多次
   python main.py --config configs\django_real_mmoe.yaml --seed 42
   python main.py --config configs\django_real_mmoe.yaml --seed 43
   python main.py --config configs\django_real_mmoe.yaml --seed 44
   # ... 重复5-10次
   ```

2. **交叉验证**：使用k-fold交叉验证，每个fold产生一个测量值

3. **跨项目实验**（推荐）：每个测试项目产生一个测量值

如果只有单次训练结果（每个模型只有1个值），将无法计算标准差和p值，此时只能进行简单的数值对比。

```powershell
# 推荐方法：基于跨项目实验比较两个模型（样本量充足）
python experiments\hypothesis_test.py --result_dirs outputs\cross_project_mmoe outputs\cross_project_shared --output outputs\hypothesis_test_cross_project --metric R2
python experiments\hypothesis_test.py --result_dirs outputs\cross_project_mmoe outputs\cross_project_shared --output outputs\hypothesis_test_cross_project --metric F1

# 备选方法：基于单项目结果比较（样本量有限，统计效力较弱）
python experiments\hypothesis_test.py --result_dirs outputs\django_real_mmoe outputs\django_real_shared --output outputs\hypothesis_test --metric R2
python experiments\hypothesis_test.py --result_dirs outputs\django_real_mmoe outputs\django_real_shared --output outputs\hypothesis_test --metric F1
```

**重要说明**：
- **推荐使用跨项目比较**：5个测试项目 = 5个样本，统计检验更可靠
- 单项目比较通常样本量不足（n=1），无法进行有效的统计检验
- 跨项目比较不仅评估模型性能，还评估跨项目泛化能力

支持的指标：
- 回归: `R2`, `MAE`, `MSE`, `RMSE`
- 分类: `Accuracy`, `Precision`, `Recall`, `F1`, `F1_macro`

## 实验流程

### 完整实验流程（推荐）

```powershell
# 1. 首先确保已经训练了基础模型（可选，仅用于单项目比较）
python main.py --config configs\django_real_mmoe.yaml
python main.py --config configs\django_real_shared.yaml

# 2. 运行所有level2实验（包括跨项目模型比较和假设检验）
python experiments\run_all_experiments.py --base_config configs\django_real_mmoe.yaml

# 3. 查看结果
# - 消融实验结果: outputs/ablation_mmoe/, outputs/ablation_shared/
# - 跨项目泛化演示: outputs/cross_project/
# - 跨项目模型比较: outputs/cross_project_mmoe/, outputs/cross_project_shared/
# - 假设检验（单项目）: outputs/hypothesis_test/
# - 假设检验（跨项目）: outputs/hypothesis_test_cross_project/
# - 可视化图表: outputs/figures_level2/
```

## 输出结果说明

### 消融实验输出

每个消融实验会生成：
- `{ablation_type}.json`: 该消融实验的详细结果
- `ablation_summary.json`: 所有消融实验的汇总

结果包含：
- 原始配置和修改后的配置
- 测试集上的回归和分类指标
- 训练历史

### 跨项目实验输出

**单个实验结果**：
- `train_{train_projects}_test_{test_project}.json`: 每个跨项目实验的详细结果
  - 示例：`train_django_react_moby_opencv_test_tensorflow.json`
  - 包含：训练项目列表、测试项目、回归指标、分类指标、训练历史

**汇总统计**：
- `cross_project_summary.json`: 所有跨项目实验的汇总
  - 包含：各指标的平均值和标准差
  - 用于评估模型的跨项目泛化能力

**目录结构示例**：
```
outputs/
├── cross_project_mmoe/              # MMoE跨项目结果
│   ├── train_react_tensorflow_moby_opencv_test_django.json
│   ├── train_django_tensorflow_moby_opencv_test_react.json
│   ├── train_django_react_moby_opencv_test_tensorflow.json
│   ├── train_django_react_tensorflow_opencv_test_moby.json
│   ├── train_django_react_tensorflow_moby_test_opencv.json
│   └── cross_project_summary.json
└── cross_project_shared/            # Shared-Bottom跨项目结果
    ├── train_react_tensorflow_moby_opencv_test_django.json
    ├── ... (结构同上)
    └── cross_project_summary.json
```

### 假设检验输出

**两模型比较**（本实验使用）：
- `comparison_{metric}.json`: 完整统计检验结果
  - 包含：Wilcoxon signed-rank test、Paired t-test、Cohen's d
  - 描述性统计：均值、标准差、中位数、最小/最大值
  - 每个样本的详细数据和差异
- `comparison_{metric}.png`: 4个子图的可视化
  - 箱线图：两模型分布对比
  - 散点图：配对样本对比（对角线表示性能相等）
  - 直方图：差异分布
  - 统计摘要表：p值、效应量、结论

**多模型比较**（3+模型时使用）：
- `hypothesis_test_{metric}.json`: Friedman检验的详细结果
- `nemenyi_{metric}.png`: Nemenyi post-hoc检验的可视化热力图

结果包括：
- 统计检验的统计量和p值
- 各模型的平均秩次（多模型）或描述性统计（两模型）
- 显著差异的判断和解释
- 效应量（Cohen's d，两模型时）

**输出目录**：
```
outputs/hypothesis_test_cross_project/
├── comparison_R2.json        # 回归R2比较结果
├── comparison_R2.png          # 回归R2可视化
├── comparison_F1.json         # 分类F1比较结果
└── comparison_F1.png          # 分类F1可视化
```

## 依赖包

除了项目基础依赖外，假设检验需要额外的包：

```powershell
pip install scikit-posthocs
```

如果未安装，运行假设检验时会提示安装。

## 注意事项

1. **数据要求**: 
   - 消融实验使用配置文件中指定的数据集
   - 跨项目实验需要engineered目录下有多个项目的数据文件

2. **计算时间**:
   - 消融实验：每个消融类型需要完整训练一次模型
   - 跨项目实验：对N个项目，需要训练N个模型（每次使用N-1个项目训练，在1个项目测试）
   - 假设检验：仅分析已有结果，计算很快

3. **结果解读**:
   - 消融实验：如果移除某组件后性能下降，说明该组件重要
   - 跨项目实验：标准差越小说明泛化能力越好
   - 假设检验：
     - p < 0.05表示模型间存在显著差异（统计显著性）
     - Cohen's d 表示实际效应大小（实际意义）
     - 需要同时考虑统计显著性和实际意义
   - **样本不足警告**：如果每个模型只有1个测量值，无法进行统计检验，建议：
     - 使用不同随机种子多次训练（5-10次）
     - 或使用跨项目实验（每个项目一个样本）
     - 或使用交叉验证

## 示例工作流

```powershell
# 完整的level2实验工作流

# Step 1: 准备环境
pip install -r requirements.txt
pip install scikit-posthocs

# Step 2: 训练基础模型（如果还没有）
python main.py --config configs\django_mmoe.yaml
python main.py --config configs\django_shared.yaml

# Step 3: 运行所有实验
python experiments\run_all_experiments.py

# Step 4: 或者分步运行
python experiments\ablation_study.py --config configs\django_mmoe.yaml --output outputs\ablation_mmoe --ablation all
python experiments\cross_project.py --config configs\django_mmoe.yaml --output outputs\cross_project --projects django react tensorflow
python experiments\hypothesis_test.py --result_dirs outputs\django_mmoe outputs\django_shared --output outputs\hypothesis_test --metric R2

# Step 5: 查看结果
# 结果保存在outputs目录下的对应子目录中
```

## 故障排除

### 问题1: ModuleNotFoundError: No module named 'scikit_posthocs'

```powershell
pip install scikit-posthocs
```

### 问题2: 跨项目实验找不到数据文件

确保engineered目录下有对应的数据文件，例如：
- engineered/django_engineered.xlsx
- engineered/react_engineered.xlsx
- engineered/tensorflow_engineered.xlsx

### 问题3: 假设检验找不到结果文件

确保先运行了基础模型训练或跨项目实验，生成了results.json文件。

### 问题4: 假设检验显示"样本数量不足"

**原因**：每个模型只有1个测量值（单次训练结果），无法计算标准差和进行统计检验。

**解决方案**：

1. **使用不同随机种子多次训练**（推荐5-10次）：
   ```powershell
   # 修改配置文件中的seed，或使用命令行参数
   python main.py --config configs\django_real_mmoe.yaml --seed 42 --output outputs\django_real_mmoe_seed42
   python main.py --config configs\django_real_mmoe.yaml --seed 43 --output outputs\django_real_mmoe_seed43
   # ... 重复多次
   
   # 然后比较多次运行的结果
   python experiments\hypothesis_test.py --result_dirs outputs\django_real_mmoe_seed42 outputs\django_real_mmoe_seed43 ... --output outputs\hypothesis_test --metric R2
   ```

2. **使用跨项目实验**（推荐，自动产生多个样本）：
   ```powershell
   # 每个测试项目产生一个样本点
   python experiments\cross_project.py --config configs\django_real_mmoe.yaml --output outputs\cross_project --projects django react tensorflow moby opencv
   
   # 基于跨项目结果进行假设检验
   python experiments\hypothesis_test.py --result_dirs outputs\cross_project --output outputs\hypothesis_test --metric R2 --cross_project
   ```

3. **使用交叉验证**：实现k-fold交叉验证，每个fold产生一个测量值

**为什么需要多个样本**：
- 统计检验需要计算标准差来评估变异性
- 单个值无法判断差异是偶然还是系统性的
- 多个样本可以提供更可靠的结论

## 更多信息

详细的实验设计和结果分析请参见：
- 项目根目录的 `REPORT.md`（实验报告）
- 各实验输出目录下的JSON结果文件
