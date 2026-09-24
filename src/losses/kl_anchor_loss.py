"""L_anchor: KL 散度锚定损失。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KLAnchorLoss(nn.Module):

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        online_pooled: torch.Tensor,
        frozen_pooled: torch.Tensor,
    ) -> torch.Tensor:
        online_norm = F.normalize(online_pooled, dim=-1)
        frozen_norm = F.normalize(frozen_pooled, dim=-1)

        sim_online = torch.mm(online_norm, online_norm.t()) / self.temperature
        sim_frozen = torch.mm(frozen_norm, frozen_norm.t()) / self.temperature

        probs_online = F.log_softmax(sim_online, dim=-1)
        probs_frozen = F.softmax(sim_frozen, dim=-1)

        return F.kl_div(probs_online, probs_frozen, reduction="batchmean")
