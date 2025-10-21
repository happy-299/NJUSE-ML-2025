# PR 多任务学习项目

项目采用标准的机器学习项目结构，将代码、配置、数据和输出清晰分离：

- 模型：`models/`（基础组件、MMoE、SharedBottom 模型实现）
- 工具：`utils/`（数据处理、指标计算、可视化）
- 配置：`configs/`（实验配置文件）
- 输出：`outputs/`（训练结果和图表）
- 检查点：`checkpoints/`（保存模型权重）
- 数据：`data/`（原始和预处理数据）

## 准备环境

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 验证环境是否正确配置
python experiments\check_environment.py
```

## 数据准备

首次运行前，请先准备数据：

```powershell
# 将数据文件复制到data目录
python migrate_data.py
# 或者使用数据获取脚本
python data/get_data.py
```

## 运行训练

```powershell
# 方式A：运行示例配置（example.yaml）
python main.py --config configs\example.yaml

# 方式B：运行实际实验配置（Shared-Bottom）
python main.py --config configs\django_shared.yaml

# 方式C：运行对比实验配置（MMoE）
python main.py --config configs\django_mmoe.yaml
```

说明：
- Windows PowerShell 下参数路径可以用 `\`（推荐）或 `/` 分隔，Python 都能识别。
- 如果不传 `--config`，脚本会使用默认的 `./configs/example.yaml`。（等价于运行方式A）
- 训练过程的模型与指标会写到配置里的 `output.dir`（例如 `./outputs/django_shared`）。

为何一次训练只用一个表？
- `dataset.file` 明确指向单一数据源，保证切分在同一表内完成、实验可复现、避免跨文件泄漏。
- 如需多仓库/多文件：
	- 方式一：先合并为单表并添加 `repo` 列（推荐）：
		```powershell
		python scripts\merge_csv.py --input_dir data --output data\merged.csv
		```
		然后在配置里将 `dataset.file` 改为 `./data/merged.csv`。
	- 方式二：分别训练后再对比：为每个文件各自建一份配置，训练完成后使用 `utils/visualization.py` 中的函数汇总。

## 目录说明

- `models/base.py`：基础网络组件（嵌入层、MLP 等）
- `models/mmoe.py`：MMoE 模型实现（多专家混合网络）
- `models/shared_bottom.py`：SharedBottom 模型实现（共享底层网络）
- `models/model.py`：完整的多任务学习模型
- `utils/data_processing.py`：CSV/XLSX 读取、数值标准化、类别 embedding/one-hot、数据集切分
- `utils/metrics.py`：MAE/MSE/RMSE/R2 与 Accuracy/Precision/Recall/F1
- `utils/visualization.py`：训练历史和模型比较可视化
- `config.py`：默认配置管理
- `main.py`：训练主程序入口
- `configs/example.yaml`：训练配置示例（可复制修改）

报告与图表：
- 实验报告：`REPORT.md`
- 图表输出：`outputs/figures/`（使用 `utils/visualization.py` 生成）

可视化结果示例：
```python
# 绘制单个模型的训练历史
from utils.visualization import plot_training_history
plot_training_history("outputs/example/history.json", save_dir="outputs/figures")

# 比较多个模型
from utils.visualization import plot_model_comparison
plot_model_comparison(
    ["outputs/django_mmoe", "outputs/django_shared"],
    ["MMoE", "SharedBottom"],
    save_path="outputs/figures/model_comparison.png"
)
```

如需自定义列名或处理规则，请编辑 `configs/example.yaml` 或在 `utils/data_processing.py` 中扩展。

## Level 2 实验

Level 2 包含以下高级实验：

### 1. 模型组件消融实验
测试不同模型组件对性能的影响：

```powershell
# 运行所有消融实验
python experiments\ablation_study.py --config configs\django_mmoe.yaml --output outputs\ablation_mmoe --ablation all
```

支持的消融类型：
- `no_dropout`: 移除Dropout层
- `shallow_tower`: 浅层任务塔
- `small_embedding`: 小embedding维度
- `fewer_experts`: 减少专家数量（MMoE）
- `single_expert`: 单专家（MMoE）

### 2. 跨项目泛化性实验
评估模型在不同项目间的泛化能力：

```powershell
# 使用3个项目进行跨项目实验
python experiments\cross_project.py --config configs\django_mmoe.yaml --output outputs\cross_project --projects django react tensorflow
```

### 3. 假设检验
使用Friedman test和Nemenyi post-hoc test严格比较模型性能：

```powershell
# 比较两个模型
python experiments\hypothesis_test.py --result_dirs outputs\django_mmoe outputs\django_shared --output outputs\hypothesis_test --metric R2

# 基于跨项目结果进行检验
python experiments\hypothesis_test.py --result_dirs outputs\cross_project --output outputs\hypothesis_test --metric R2 --cross_project
```

### 运行所有Level 2实验

```powershell
# 一键运行所有实验
python experiments\run_all_experiments.py --base_config configs\django_mmoe.yaml
```

详细说明请参见 `experiments/README.md`。