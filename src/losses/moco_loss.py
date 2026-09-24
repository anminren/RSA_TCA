"""L_moco: MoCo-v3 对称 InfoNCE 对比损失。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoLoss(nn.Module):

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        logits_qk = torch.mm(query, key.t()) / self.temperature
        logits_kq = torch.mm(key, query.t()) / self.temperature

        batch_size = query.size(0)
        labels = torch.arange(batch_size, device=query.device)

        loss_qk = F.cross_entropy(logits_qk, labels)
        loss_kq = F.cross_entropy(logits_kq, labels)

        return (loss_qk + loss_kq) / 2.0
