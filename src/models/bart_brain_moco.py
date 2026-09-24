"""BART Brain-MoCo 完整框架。

将 BartBackbone + 三种投影头组合为 MoCo-v3 风格的双编码器框架：
- Online encoder: BartBackbone (全量微调 encoder) + BrainProjectionHead + ContrastiveProjectionHead + PredictionHead
- Momentum encoder: BartBackbone (frozen deepcopy) + ContrastiveProjectionHead (frozen deepcopy)

关键差异（相比 Llama 版本）：
- 不使用 LoRA，直接全量微调 encoder
- 冻结 decoder 的所有参数
"""

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn

from .bart_backbone import BartBackbone
from .projection_heads import (
    BrainProjectionHead,
    ContrastiveProjectionHead,
    PredictionHead,
)


class BartBrainMoCo(nn.Module):

    def __init__(
        self,
        model_name: str,
        n_channels: int = 32,
        pooling: str = "last",
        brain_head_dropout: float = 0.3,
        contrastive_proj_dim: int = 256,
        ema_momentum: float = 0.999,
        enable_moco: bool = True,
        freeze_encoder: bool = False,
        frozen_anchor: bool = False,
        freeze_decoder: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.ema_momentum = ema_momentum
        self.freeze_encoder = freeze_encoder
        self.frozen_anchor = frozen_anchor

        if freeze_encoder:
            enable_moco = False

        self.enable_moco = enable_moco

        # === Online 编码器 ===
        self.online_encoder = BartBackbone(
            model_name=model_name,
            pooling=pooling,
            output_hidden_states=True,
            freeze_decoder=freeze_decoder,
            gradient_checkpointing=gradient_checkpointing,
        )
        hidden_dim = self.online_encoder.get_hidden_dim()

        if freeze_encoder:
            for param in self.online_encoder.parameters():
                param.requires_grad = False

        # === Brain 投影头 ===
        self.brain_head = BrainProjectionHead(
            hidden_dim=hidden_dim,
            n_channels=n_channels,
            dropout=brain_head_dropout,
        )

        # === MoCo 组件 ===
        if enable_moco:
            self.online_projector = ContrastiveProjectionHead(
                hidden_dim=hidden_dim,
                proj_dim=contrastive_proj_dim,
            )
            self.predictor = PredictionHead(proj_dim=contrastive_proj_dim)

            self.momentum_encoder = copy.deepcopy(self.online_encoder)
            self.momentum_projector = copy.deepcopy(self.online_projector)

            for param in self.momentum_encoder.parameters():
                param.requires_grad = False
            for param in self.momentum_projector.parameters():
                param.requires_grad = False

        # === 冻结锚定编码器 ===
        if frozen_anchor:
            self.frozen_encoder = BartBackbone(
                model_name=model_name,
                pooling=pooling,
                output_hidden_states=False,
                freeze_decoder=freeze_decoder,
                gradient_checkpointing=False,
            )
            for param in self.frozen_encoder.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def _ema_update(self) -> None:
        if not self.enable_moco:
            return

        m = self.ema_momentum
        for param_q, param_k in zip(
            self.online_encoder.parameters(), self.momentum_encoder.parameters()
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)

        for param_q, param_k in zip(
            self.online_projector.parameters(), self.momentum_projector.parameters()
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_word_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # === Online 编码器 ===
        online_out = self.online_encoder(input_ids, attention_mask, target_word_mask)
        target_word_hidden = online_out["target_word_hidden"]
        online_pooled = online_out["pooled_embedding"]

        brain_pred = self.brain_head(target_word_hidden)

        result = {
            "brain_pred": brain_pred,
            "online_pooled": online_pooled,
            "all_hidden_states": online_out["all_hidden_states"],
        }

        # === MoCo 组件 ===
        if self.enable_moco:
            online_proj = self.online_projector(online_pooled)
            query = self.predictor(online_proj)

            with torch.no_grad():
                momentum_out = self.momentum_encoder(
                    input_ids, attention_mask, target_word_mask
                )
                momentum_pooled = momentum_out["pooled_embedding"]
                key = self.momentum_projector(momentum_pooled)

            result["query"] = query
            result["key"] = key.detach()
            result["momentum_pooled"] = momentum_pooled.detach()

        # === 冻结锚定编码器 ===
        if self.frozen_anchor:
            with torch.no_grad():
                frozen_out = self.frozen_encoder(
                    input_ids, attention_mask, target_word_mask
                )
                frozen_pooled = frozen_out["pooled_embedding"]
            result["frozen_pooled"] = frozen_pooled.detach()

        return result

    def get_trainable_param_groups(
        self,
        lr_encoder: float = 1e-5,
        lr_brain_head: float = 1e-4,
        lr_contrastive_heads: float = 1e-4,
        weight_decay_encoder: float = 1e-4,
        weight_decay_head: float = 2e-4,
    ) -> list:
        encoder_params = []
        for name, param in self.online_encoder.named_parameters():
            if param.requires_grad:
                encoder_params.append(param)

        brain_params = list(self.brain_head.parameters())

        contrastive_params = []
        if self.enable_moco:
            contrastive_params.extend(self.online_projector.parameters())
            contrastive_params.extend(self.predictor.parameters())

        param_groups = []

        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": lr_encoder,
                "weight_decay": weight_decay_encoder,
                "name": "encoder",
            })

        param_groups.append({
            "params": brain_params,
            "lr": lr_brain_head,
            "weight_decay": weight_decay_head,
            "name": "brain_head",
        })

        if contrastive_params:
            param_groups.append({
                "params": contrastive_params,
                "lr": lr_contrastive_heads,
                "weight_decay": weight_decay_head,
                "name": "contrastive_heads",
            })

        return param_groups
