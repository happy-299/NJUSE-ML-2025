from typing import List

import torch
import torch.nn as nn

from models.base import BottomMLP


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], dropout: float = 0.0):
        """
        专家模块
        
        Args:
            input_dim: 输入维度
            hidden: 隐藏层节点数列表
            dropout: Dropout率
        """
        super().__init__()
        dims = [input_dim] + hidden
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden[-1] if hidden else input_dim

    def forward(self, x):
        return self.net(x)


class Gate(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        """
        门控网络
        
        Args:
            input_dim: 输入维度
            num_experts: 专家数量
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE) 模型
    """
    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 expert_hidden: List[int],
                 tower_hidden_reg: List[int],
                 tower_hidden_cls: List[int],
                 dropout: float = 0.0):
        super().__init__()
        self.experts = nn.ModuleList([Expert(input_dim, expert_hidden, dropout) for _ in range(num_experts)])
        self.gate_reg = Gate(input_dim, num_experts)
        self.gate_cls = Gate(input_dim, num_experts)

        expert_out_dim = self.experts[0].output_dim if self.experts else input_dim
        self.tower_reg = BottomMLP(expert_out_dim, tower_hidden_reg, dropout)
        self.tower_cls = BottomMLP(expert_out_dim, tower_hidden_cls, dropout)

        self.head_reg = nn.Linear(self.tower_reg.output_dim, 1)
        self.head_cls = nn.Linear(self.tower_cls.output_dim, 1)

    def forward(self, x):
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [B, N_exp, D]
        gate_w_reg = self.gate_reg(x).unsqueeze(-1)  # [B, N_exp, 1]
        gate_w_cls = self.gate_cls(x).unsqueeze(-1)
        mix_reg = torch.sum(gate_w_reg * expert_outputs, dim=1)  # [B, D]
        mix_cls = torch.sum(gate_w_cls * expert_outputs, dim=1)

        tr = self.tower_reg(mix_reg)
        tc = self.tower_cls(mix_cls)

        y_reg = self.head_reg(tr)
        y_cls = torch.sigmoid(self.head_cls(tc))
        return y_reg, y_cls