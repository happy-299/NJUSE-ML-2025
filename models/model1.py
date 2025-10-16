# models/model1.py
# 示例模型定义：可替换为 PyTorch / TensorFlow 实现

from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_dim: int = 10
    hidden_dim: int = 64
    output_dim: int = 1


class SimpleModel:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def forward(self, x):
        """示例前向函数"""
        # 真实模型请使用框架（PyTorch/TensorFlow）实现
        return x
