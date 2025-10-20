from typing import List

import torch
import torch.nn as nn

from models.base import BottomMLP


class SharedBottom(nn.Module):
    """
    SharedBottom 模型 - 共享底层网络的多任务学习模型
    """
    def __init__(self,
                 input_dim: int,
                 bottom_hidden: List[int],
                 tower_hidden_reg: List[int],
                 tower_hidden_cls: List[int],
                 dropout: float = 0.0):
        super().__init__()
        self.bottom = BottomMLP(input_dim, bottom_hidden, dropout)
        bdim = self.bottom.output_dim
        self.tower_reg = BottomMLP(bdim, tower_hidden_reg, dropout)
        self.tower_cls = BottomMLP(bdim, tower_hidden_cls, dropout)
        self.head_reg = nn.Linear(self.tower_reg.output_dim, 1)
        self.head_cls = nn.Linear(self.tower_cls.output_dim, 1)

    def forward(self, x):
        b = self.bottom(x)
        tr = self.tower_reg(b)
        tc = self.tower_cls(b)
        y_reg = self.head_reg(tr)
        y_cls = torch.sigmoid(self.head_cls(tc))
        return y_reg, y_cls