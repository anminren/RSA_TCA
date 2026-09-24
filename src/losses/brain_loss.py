"""L_brain: 脑信号预测损失 (MSE)。"""

import torch
import torch.nn as nn


class BrainLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self, brain_pred: torch.Tensor, eeg_target: torch.Tensor
    ) -> torch.Tensor:
        return self.mse(brain_pred, eeg_target)
