"""Seq2Seq encoder backbone used by the EEG tuning models.

The historical class name is kept as ``BartBackbone`` so existing training code
and checkpoints continue to work. Internally it now supports BART, Pegasus, and
Pegasus-X style encoder-decoder models through HuggingFace Auto classes.
"""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = TaskType = get_peft_model = None


class BartBackbone(nn.Module):
    """Encoder wrapper for BART/Pegasus-like seq2seq models.

    Forward returns three values:
    1. target_word_hidden: mean hidden state over the target word sub-tokens
    2. pooled_embedding: sentence-level embedding, defaulting to the last token
    3. all_hidden_states: encoder hidden states for analysis/evaluation
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "last",
        output_hidden_states: bool = True,
        freeze_decoder: bool = True,
        gradient_checkpointing: bool = False,
        lora_config: Optional[dict] = None,
    ):
        super().__init__()
        self.pooling = pooling
        self.output_hidden_states = output_hidden_states
        self.freeze_decoder = freeze_decoder
        self.lora_config = lora_config or {}

        is_local = Path(model_name).is_dir()
        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=output_hidden_states,
            local_files_only=is_local,
        )
        self.model_type = getattr(self.model.config, "model_type", "unknown")

        if self.lora_config.get("enabled", False):
            if get_peft_model is None:
                raise ImportError("peft is required for LoRA training")
            target_modules = list(self.lora_config.get("target_modules", ["q_proj", "v_proj"]))
            bias = self.lora_config.get("bias", "none") or "none"
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=int(self.lora_config.get("r", 8)),
                lora_alpha=int(self.lora_config.get("alpha", 16)),
                lora_dropout=float(self.lora_config.get("dropout", 0.05)),
                target_modules=target_modules,
                bias=str(bias),
            )
            self.model = get_peft_model(self.model, peft_config)

        encoder = self._get_encoder()
        if gradient_checkpointing:
            # Non-reentrant checkpointing avoids the Pegasus case where integer
            # input IDs make checkpointed blocks report that gradients are None.
            target = self.model if hasattr(self.model, "gradient_checkpointing_enable") else encoder
            if hasattr(target, "gradient_checkpointing_enable"):
                try:
                    target.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    target.gradient_checkpointing_enable()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False

        self.hidden_dim = (
            getattr(self.model.config, "hidden_size", None)
            or getattr(self.model.config, "d_model", None)
        )
        if self.hidden_dim is None:
            raise ValueError(
                f"Cannot infer hidden dim from model config: {self.model.config}"
            )

        if freeze_decoder:
            self._freeze_decoder()

    def _get_encoder(self):
        if hasattr(self.model, "get_encoder"):
            try:
                return self.model.get_encoder()
            except Exception:
                pass
        if hasattr(self.model, "encoder"):
            return self.model.encoder
        if hasattr(self.model, "model") and hasattr(self.model.model, "encoder"):
            return self.model.model.encoder
        if hasattr(self.model, "base_model"):
            base = self.model.base_model
            if hasattr(base, "model") and hasattr(base.model, "encoder"):
                return base.model.encoder
            if hasattr(base, "model") and hasattr(base.model, "model") and hasattr(base.model.model, "encoder"):
                return base.model.model.encoder
        raise AttributeError(f"Model {type(self.model).__name__} has no encoder")

    def _freeze_decoder(self) -> None:
        decoder = None
        if hasattr(self.model, "get_decoder"):
            try:
                decoder = self.model.get_decoder()
            except Exception:
                decoder = None
        if decoder is None and hasattr(self.model, "decoder"):
            decoder = self.model.decoder
        if decoder is None and hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):
            decoder = self.model.model.decoder
        if decoder is None and hasattr(self.model, "base_model"):
            base = self.model.base_model
            if hasattr(base, "model") and hasattr(base.model, "decoder"):
                decoder = base.model.decoder
            elif hasattr(base, "model") and hasattr(base.model, "model") and hasattr(base.model.model, "decoder"):
                decoder = base.model.model.decoder

        if decoder is not None:
            for param in decoder.parameters():
                param.requires_grad = False
            if hasattr(decoder, "embed_positions"):
                for param in decoder.embed_positions.parameters():
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_word_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        encoder = self._get_encoder()
        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.output_hidden_states,
            return_dict=True,
        )

        last_hidden = encoder_outputs.last_hidden_state
        target_word_hidden = self._masked_mean_pool(last_hidden, target_word_mask)
        pooled_embedding = self._pool(last_hidden, attention_mask)

        all_hidden_states = None
        if self.output_hidden_states and encoder_outputs.hidden_states is not None:
            all_hidden_states = encoder_outputs.hidden_states

        return {
            "target_word_hidden": target_word_hidden,
            "pooled_embedding": pooled_embedding,
            "all_hidden_states": all_hidden_states,
        }

    def _masked_mean_pool(
        self, hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask_expanded = mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        return sum_hidden / count

    def _pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.pooling == "last":
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            return hidden_states[
                torch.arange(batch_size, device=hidden_states.device), seq_lengths
            ]
        if self.pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            return sum_hidden / count
        raise ValueError(f"Unknown pooling mode: {self.pooling}")

    def get_hidden_dim(self) -> int:
        return self.hidden_dim

    @staticmethod
    def load_tokenizer(model_name: str):
        is_local = Path(model_name).is_dir()
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=is_local)
        model_type = getattr(cfg, "model_type", "")
        use_fast = not model_type.startswith("pegasus")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=is_local,
            use_fast=use_fast,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
