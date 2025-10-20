# 配置文件

import os
from typing import Dict, Any

# 路径配置
DATA_DIR = "./data"
CHECKPOINTS_DIR = "./checkpoints"
OUTPUTS_DIR = "./outputs"

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# 默认训练参数
DEFAULT_TRAINING_CONFIG = {
    "batch_size": 256,
    "num_workers": 0,
    "epochs": 50,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "patience": 5,
    "loss_weights": {
        "reg": 1.0,
        "cls": 1.0
    }
}

# 默认模型参数
DEFAULT_MODEL_CONFIG = {
    "type": "mmoe",  # 'mmoe' 或 'shared_bottom'
    "embedding_dim": 16,
    "bottom_mlp": [128, 64],
    "experts": 4,
    "expert_hidden": [64],
    "tower_hidden_reg": [64, 32],
    "tower_hidden_cls": [64, 32],
    "dropout": 0.1
}

# 默认数据集参数
DEFAULT_DATASET_CONFIG = {
    "file": os.path.join(DATA_DIR, "synth.csv"),
    "sheet_name": None,
    "numeric_features": None,  # 自动检测
    "categorical_features": None,  # 自动检测
    "target_reg": "pr_close_time_hours",
    "target_cls": "pr_merged",
    "one_hot_categorical": False,
    "split": {
        "test_size": 0.1,
        "val_size": 0.1,
        "stratify_by": "pr_merged"
    }
}

# 默认输出参数
DEFAULT_OUTPUT_CONFIG = {
    "dir": OUTPUTS_DIR
}

# 默认随机种子
DEFAULT_SEED = 42

# 合并默认配置
def get_default_config() -> Dict[str, Any]:
    """获取默认配置字典"""
    return {
        "seed": DEFAULT_SEED,
        "dataset": DEFAULT_DATASET_CONFIG,
        "model": DEFAULT_MODEL_CONFIG,
        "training": DEFAULT_TRAINING_CONFIG,
        "output": DEFAULT_OUTPUT_CONFIG
    }

# 加载自定义配置文件并与默认值合并
def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件并与默认配置合并"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        custom_config = yaml.safe_load(f)
    
    default_config = get_default_config()
    
    # 合并配置（简单深度合并）
    for key, default_value in default_config.items():
        if key not in custom_config:
            custom_config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(custom_config[key], dict):
            for subkey, subvalue in default_value.items():
                if subkey not in custom_config[key]:
                    custom_config[key][subkey] = subvalue
    
    return custom_config