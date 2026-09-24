"""组合损失：L = λ_brain · L_brain + λ_moco · L_moco + λ_anchor · L_anchor"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .brain_loss import BrainLoss
from .moco_loss import MoCoLoss
from .anchor_loss import AnchorLoss
from .cosine_anchor_loss import CosineAnchorLoss
from .kl_anchor_loss import KLAnchorLoss


class CombinedLoss(nn.Module):

    def __init__(
        self,
        lambda_brain: float = 1.0,
        lambda_moco: float = 0.1,
        lambda_anchor: float = 0.05,
        moco_temperature: float = 0.2,
        anchor_type: str = "momentum",
        kl_temperature: float = 0.1,
    ):
        super().__init__()
        self.lambda_brain = lambda_brain
        self.lambda_moco = lambda_moco
        self.lambda_anchor = lambda_anchor
        self.anchor_type = anchor_type

        self.brain_loss = BrainLoss()

        if lambda_moco > 0:
            self.moco_loss = MoCoLoss(temperature=moco_temperature)

        if lambda_anchor > 0 and anchor_type != "none":
            if anchor_type == "momentum":
                self.anchor_loss = AnchorLoss()
            elif anchor_type == "cosine_only":
                self.anchor_loss = CosineAnchorLoss()
            elif anchor_type == "kl":
                self.anchor_loss = KLAnchorLoss(temperature=kl_temperature)
            else:
                raise ValueError(f"Unknown anchor_type: {anchor_type}")

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        eeg_target: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        result = {}
        total_loss = torch.tensor(0.0, device=eeg_target.device)

        if self.lambda_brain > 0:
            loss_brain = self.brain_loss(model_output["brain_pred"], eeg_target)
            result["loss_brain"] = loss_brain
            total_loss = total_loss + self.lambda_brain * loss_brain
        else:
            result["loss_brain"] = torch.tensor(0.0, device=eeg_target.device)

        if self.lambda_moco > 0 and "query" in model_output:
            loss_moco = self.moco_loss(model_output["query"], model_output["key"])
            result["loss_moco"] = loss_moco
            total_loss = total_loss + self.lambda_moco * loss_moco
        else:
            result["loss_moco"] = torch.tensor(0.0, device=eeg_target.device)

        if self.lambda_anchor > 0 and self.anchor_type != "none":
            loss_anchor = self._compute_anchor_loss(model_output)
            if loss_anchor is not None:
                result["loss_anchor"] = loss_anchor
                total_loss = total_loss + self.lambda_anchor * loss_anchor
            else:
                result["loss_anchor"] = torch.tensor(0.0, device=eeg_target.device)
        else:
            result["loss_anchor"] = torch.tensor(0.0, device=eeg_target.device)

        result["loss"] = total_loss
        return result

    def _compute_anchor_loss(
        self,
        model_output: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.anchor_type == "momentum":
            if "momentum_pooled" in model_output:
                return self.anchor_loss(
                    model_output["online_pooled"], model_output["momentum_pooled"]
                )
        elif self.anchor_type in ("cosine_only", "kl"):
            if "frozen_pooled" in model_output:
                return self.anchor_loss(
                    model_output["online_pooled"], model_output["frozen_pooled"]
                )
        return None
