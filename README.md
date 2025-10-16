# NJUSE-ML-2025
NJUSE-ML-2025 is the project repository for the Machine Learning course in the School of Software Engineering at Nanjing University in 2025.


## 项目目录说明

- **checkpoints/** 或 **data/**
  - 保存训练好的模型 checkpoint 或原始/预处理后的数据文件。
  - 例如：在训练过程中将模型和最佳权重保存到 `checkpoints/`，将下载的数据或中间处理结果放到 `data/`。

- **data/get_data.py**
  - 数据获取和预处理脚本（示例）。

- **models/**
  - 模型定义目录，用于放置不同模型结构的实现文件，例如 `model1.py`、`model2.py` 等。
  - 每个模型文件应包含模型类和必要的配置说明。

- **utils/**
  - 工具函数和辅助模块，如可视化 (`visualization.py`)、计算评估指标 (`calculation.py`) 等。

- **config.py**
  - 配置文件，存放默认参数、路径和超参数。

- **main.py**
  - 训练与测试程序的入口，解析命令行参数并调用相应的流程。

- **requirements.txt**
  - 列出项目依赖的第三方库，便于创建虚拟环境和安装依赖。

## 使用示例

- 训练：
  ```bash
  python main.py --mode train
  ```

- 测试：
  ```bash
  python main.py --mode test
  ```


## 建议

- 在开始训练前，请将数据放到 `data/` 下，或在 `data/get_data.py` 中实现数据下载逻辑。
- 将训练过程中产生的模型保存到 `checkpoints/`，并加入 .gitignore（如果需要）。
- 根据实际框架（PyTorch/TensorFlow）替换 `models/` 中的示例实现。
