from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import EmbeddingBlock, BottomMLP
from models.mmoe import MMoE
from models.shared_bottom import SharedBottom


class MTLModel(nn.Module):
    """
    多任务学习模型 - 支持 MMoE 和 SharedBottom 两种架构
    """

    def __init__(self,
                 numeric_dim: int,
                 cat_cardinalities: List[int],
                 cat_onehot_dim: int,
                 embedding_dim: int,
                 bottom_mlp: List[int],
                 model_type: str,
                 experts: int,
                 expert_hidden: List[int],
                 tower_hidden_reg: List[int],
                 tower_hidden_cls: List[int],
                 dropout: float = 0.0):
        super().__init__()
        self.use_one_hot = cat_onehot_dim > 0
        self.embedding = EmbeddingBlock(cat_cardinalities, embedding_dim,
                                        dropout)
        fused_input_dim = numeric_dim + (cat_onehot_dim if self.use_one_hot
                                         else self.embedding.output_dim)

        self.wide = nn.Linear(fused_input_dim,
                              32) if fused_input_dim > 0 else None
        self.deep_input = BottomMLP(fused_input_dim, bottom_mlp,
                                    dropout) if bottom_mlp else None
        deep_out_dim = self.deep_input.output_dim if self.deep_input else fused_input_dim
        combined_dim = deep_out_dim + (32 if self.wide is not None else 0)

        if model_type.lower() == 'mmoe':
            self.core = MMoE(combined_dim, experts, expert_hidden,
                             tower_hidden_reg, tower_hidden_cls, dropout)
        elif model_type.lower() == 'shared_bottom':
            self.core = SharedBottom(combined_dim, [combined_dim],
                                     tower_hidden_reg, tower_hidden_cls,
                                     dropout)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def forward(self,
                x_num: Optional[torch.Tensor],
                x_cat: Optional[torch.Tensor],
                x_cat_onehot: Optional[torch.Tensor] = None):
        parts = []
        if x_num is not None and x_num.numel() > 0:
            parts.append(x_num)
        # 兼容：如果未显式传入 onehot，但 x_cat 是浮点张量，则认为已是 onehot
        if x_cat_onehot is not None and x_cat_onehot.numel() > 0:
            parts.append(x_cat_onehot)
        elif x_cat is not None and x_cat.numel(
        ) > 0 and x_cat.dtype.is_floating_point:
            parts.append(x_cat)
        else:
            emb = self.embedding(x_cat)
            if emb is not None and emb.numel() > 0:
                parts.append(emb)
        if parts:
            x = torch.cat(parts, dim=-1)
        else:
            # create an empty tensor on the same device as model parameters
            dev = next(self.parameters()).device if any(
                True for _ in self.parameters()) else torch.device('cpu')
            x = torch.empty((0, ), device=dev)

        if self.wide is not None:
            wide_out = F.relu(self.wide(x))
            if self.deep_input is not None:
                deep_out = self.deep_input(x)
                x_all = torch.cat([deep_out, wide_out], dim=-1)
            else:
                x_all = torch.cat([x, wide_out], dim=-1)
        else:
            x_all = self.deep_input(x) if self.deep_input is not None else x

        y_reg, y_cls = self.core(x_all)
        return y_reg, y_cls
