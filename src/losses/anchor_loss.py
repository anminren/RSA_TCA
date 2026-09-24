"""L_anchor: Embedding 空间锚定损失 (momentum)。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        online_pooled: torch.Tensor,
        momentum_pooled: torch.Tensor,
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(online_pooled, momentum_pooled, dim=-1)
        return (1.0 - cos_sim).mean()
