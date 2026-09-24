"""BART 简单微调实现。

结构: BartEncoder (全量微调) + Linear Head + MSE Loss。
类似 ECoG-Tuning-main 中的 WhisperLinear 简单实现。
冻结 decoder，只训练 encoder 和 brain projection head。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .bart_backbone import BartBackbone
from .projection_heads import BrainProjectionHead


class BartSimple(nn.Module):
    """BART 简单脑信号对齐模型。

    Forward 返回:
        brain_pred: (B, n_channels) 脑信号预测
        pooled_embedding: (B, hidden_dim) 句子级表示
        all_hidden_states: encoder 各层 hidden states
    """

    def __init__(
        self,
        model_name: str,
        n_channels: int = 32,
        pooling: str = "last",
        brain_head_dropout: float = 0.3,
        freeze_decoder: bool = True,
        gradient_checkpointing: bool = False,
        output_hidden_states: bool = True,
        lora_config: Optional[dict] = None,
    ):
        super().__init__()
        self.encoder = BartBackbone(
            model_name=model_name,
            pooling=pooling,
            output_hidden_states=output_hidden_states,
            freeze_decoder=freeze_decoder,
            gradient_checkpointing=gradient_checkpointing,
            lora_config=lora_config,
        )
        hidden_dim = self.encoder.get_hidden_dim()

        self.brain_head = BrainProjectionHead(
            hidden_dim=hidden_dim,
            n_channels=n_channels,
            dropout=brain_head_dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_word_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        enc_out = self.encoder(input_ids, attention_mask, target_word_mask)
        target_word_hidden = enc_out["target_word_hidden"]
        brain_pred = self.brain_head(target_word_hidden)

        return {
            "brain_pred": brain_pred,
            "pooled_embedding": enc_out["pooled_embedding"],
            "all_hidden_states": enc_out["all_hidden_states"],
        }

    def get_trainable_param_groups(
        self,
        lr_encoder: float = 1e-5,
        lr_brain_head: float = 1e-4,
        lr_contrastive_heads: float = 1e-4,
        weight_decay_encoder: float = 1e-4,
        weight_decay_head: float = 2e-4,
    ) -> list:
        encoder_params = [
            p for p in self.encoder.parameters() if p.requires_grad
        ]
        head_params = list(self.brain_head.parameters())

        param_groups = []
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": lr_encoder,
                "weight_decay": weight_decay_encoder,
                "name": "encoder",
            })
        param_groups.append({
            "params": head_params,
            "lr": lr_brain_head,
            "weight_decay": weight_decay_head,
            "name": "brain_head",
        })
        return param_groups
