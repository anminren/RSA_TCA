"""投影头模块。

Brain-MoCo 框架中使用的三种投影头：
1. BrainProjectionHead: 将目标词 hidden state 映射到 EEG 通道空间
2. ContrastiveProjectionHead: 将 pooled embedding 映射到对比学习空间
3. PredictionHead: MoCo-v3 的非对称 prediction head
"""

import torch
import torch.nn as nn


class BrainProjectionHead(nn.Module):
    """脑信号预测投影头。"""

    def __init__(self, hidden_dim: int, n_channels: int, dropout: float = 0.3):
        super().__init__()
        intermediate_dim = min(hidden_dim // 4, 256)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_dim, n_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ContrastiveProjectionHead(nn.Module):
    """对比学习投影头。"""

    def __init__(self, hidden_dim: int, proj_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PredictionHead(nn.Module):
    """预测头（仅 online 编码器使用）。"""

    def __init__(self, proj_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
