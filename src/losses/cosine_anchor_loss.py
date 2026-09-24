"""L_anchor: 与冻结编码器的余弦相似度锚定损失。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineAnchorLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        online_pooled: torch.Tensor,
        frozen_pooled: torch.Tensor,
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(online_pooled, frozen_pooled, dim=-1)
        return (1.0 - cos_sim).mean()
