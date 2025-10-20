from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingBlock(nn.Module):
    def __init__(self, cat_cardinalities: List[int], embedding_dim: int, dropout: float = 0.0):
        """
        类别特征的嵌入层
        
        Args:
            cat_cardinalities: 每个类别特征的不同值数量列表
            embedding_dim: 嵌入维度
            dropout: Dropout率
        """
        super().__init__()
        self.use_embedding = len(cat_cardinalities) > 0
        if self.use_embedding:
            self.embs = nn.ModuleList([
                nn.Embedding(card, embedding_dim) for card in cat_cardinalities
            ])
            self.dropout = nn.Dropout(dropout)
        self.output_dim = embedding_dim * len(cat_cardinalities)

    def forward(self, x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        if not self.use_embedding or x_cat is None or x_cat.numel() == 0:
            return torch.empty(x_cat.shape[0], 0, device=x_cat.device) if x_cat is not None else torch.empty(0)
        outs = []
        for i, emb in enumerate(self.embs):
            outs.append(emb(x_cat[:, i]))
        out = torch.cat(outs, dim=-1)
        return self.dropout(out)


class BottomMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: List[int], dropout: float = 0.0):
        """
        底层MLP网络
        
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