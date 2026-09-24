"""EEG-BART 训练器。"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from losses import CombinedLoss
from models import BartSimple, BartBrainMoCo
from .early_stopping import ImprovedEarlyStopping

logger = logging.getLogger("eeg_bart")


def _get_linear_warmup_decay_schedule(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


class EEGBARTTrainer:

    def __init__(
        self,
        model: Any,
        criterion: CombinedLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        save_dir: str,
        wandb_run: Optional[Any] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = wandb_run

        self.device = torch.device(
            config.experiment.device if hasattr(config.experiment, "device") else "cuda"
        )
        self.model.to(self.device)

        opt_cfg = config.training.optimizer
        param_groups = model.get_trainable_param_groups(
            lr_encoder=opt_cfg.lr_encoder,
            lr_brain_head=opt_cfg.lr_brain_head,
            lr_contrastive_heads=opt_cfg.get("lr_contrastive_heads", 1e-4),
            weight_decay_encoder=opt_cfg.weight_decay_encoder,
            weight_decay_head=opt_cfg.weight_decay_head,
        )
        self.optimizer = AdamW(param_groups)

        grad_accum = config.training.get("gradient_accumulation_steps", 1)
        steps_per_epoch = max(1, len(train_loader) // grad_accum)
        num_training_steps = steps_per_epoch * config.training.max_epochs
        warmup_ratio = config.training.scheduler.warmup_ratio
        num_warmup_steps = int(num_training_steps * warmup_ratio)
        self.scheduler = _get_linear_warmup_decay_schedule(
            self.optimizer, num_warmup_steps, num_training_steps
        )

        self.gradient_clip = config.training.gradient_clip

        self.use_amp = config.training.get("use_amp", False)
        amp_dtype_str = config.training.get("amp_dtype", "fp16")
        self.amp_dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=(self.amp_dtype == torch.float16)
            )
        else:
            self.scaler = None

        self.grad_accum_steps = config.training.get("gradient_accumulation_steps", 1)

        es_cfg = config.training.early_stopping
        self.early_stopping = ImprovedEarlyStopping(
            patience=es_cfg.patience,
            min_delta=es_cfg.min_delta,
            mode=es_cfg.mode,
            min_epochs=es_cfg.get("min_epochs", 10),
            enabled=es_cfg.get("enabled", False),
        )

        self.global_step = 0
        self.best_val_correlation = -float("inf")
        self.best_epoch = 0

    def train(self) -> Dict[str, Any]:
        max_epochs = self.config.training.max_epochs
        logger.info(f"开始训练: max_epochs={max_epochs}, device={self.device}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for epoch in range(1, max_epochs + 1):
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate(epoch)

            all_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
            all_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
            all_metrics["epoch"] = epoch

            val_corr = val_metrics.get("correlation", 0.0)
            if val_corr > self.best_val_correlation:
                self.best_val_correlation = val_corr
                self.best_epoch = epoch
                self._save_checkpoint(epoch, is_best=True)

            es_mode_value = (
                val_corr if self.early_stopping.mode == "correlation"
                else val_metrics["loss"]
            )
            if self.early_stopping(es_mode_value, epoch, train_metrics.get("correlation")):
                logger.info(
                    f"早停触发: epoch={epoch}, {self.early_stopping.get_status()}"
                )
                break

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            logger.info(f"训练 GPU 峰值显存: {peak_mb:.0f} MB")

        logger.info(
            f"训练结束: best_epoch={self.best_epoch}, "
            f"best_val_correlation={self.best_val_correlation:.4f}"
        )

        return {
            "best_epoch": self.best_epoch,
            "best_val_correlation": self.best_val_correlation,
            "final_epoch": epoch,
        }

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {"loss": [], "loss_brain": [], "loss_moco": [], "loss_anchor": []}
        all_preds = []
        all_targets = []
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_word_mask = batch["target_word_mask"].to(self.device)
            eeg_target = batch["eeg_target"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                model_output = self.model(input_ids, attention_mask, target_word_mask)
                loss_dict = self.criterion(model_output, eeg_target, model=self.model)
                loss = loss_dict["loss"] / self.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip
                        )
                    self.optimizer.step()

                self.scheduler.step()
                if hasattr(self.model, "_ema_update"):
                    self.model._ema_update()
                self.optimizer.zero_grad()
                self.global_step += 1

            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key].append(loss_dict[key].item())

            all_preds.append(model_output["brain_pred"].detach().cpu().float().numpy())
            all_targets.append(eeg_target.detach().cpu().float().numpy())

        if len(self.train_loader) % self.grad_accum_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.optimizer.step()
            self.scheduler.step()
            if hasattr(self.model, "_ema_update"):
                self.model._ema_update()
            self.optimizer.zero_grad()
            self.global_step += 1

        metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
        metrics["correlation"] = self._compute_correlation(
            np.concatenate(all_preds), np.concatenate(all_targets)
        )

        return metrics

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        epoch_losses = {"loss": [], "loss_brain": [], "loss_moco": [], "loss_anchor": []}
        all_preds = []
        all_targets = []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_word_mask = batch["target_word_mask"].to(self.device)
            eeg_target = batch["eeg_target"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                model_output = self.model(input_ids, attention_mask, target_word_mask)
                loss_dict = self.criterion(model_output, eeg_target, model=self.model)

            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key].append(loss_dict[key].item())

            all_preds.append(model_output["brain_pred"].detach().cpu().float().numpy())
            all_targets.append(eeg_target.detach().cpu().float().numpy())

        metrics = {k: np.mean(v) if v else 0.0 for k, v in epoch_losses.items()}
        metrics["correlation"] = self._compute_correlation(
            np.concatenate(all_preds), np.concatenate(all_targets)
        )

        return metrics

    @staticmethod
    def _compute_correlation(preds: np.ndarray, targets: np.ndarray) -> float:
        n_channels = preds.shape[1]
        correlations = []
        for i in range(n_channels):
            if np.std(preds[:, i]) > 0 and np.std(targets[:, i]) > 0:
                corr = np.corrcoef(preds[:, i], targets[:, i])[0, 1]
                correlations.append(corr)
        return np.mean(correlations) if correlations else 0.0

    def _get_trainable_state_dict(self) -> dict:
        trainable_names = {
            name for name, p in self.model.named_parameters() if p.requires_grad
        }
        buffer_prefixes = ("brain_head.", "online_projector.", "predictor.")
        buffer_names = {
            name
            for name, _ in self.model.named_buffers()
            if any(name.startswith(p) for p in buffer_prefixes)
        }
        save_keys = trainable_names | buffer_names
        full_state = self.model.state_dict()
        return {k: v for k, v in full_state.items() if k in save_keys}

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        trainable_state = self._get_trainable_state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainable_state,
            "best_val_correlation": self.best_val_correlation,
            "global_step": self.global_step,
            "checkpoint_type": "trainable_only",
        }

        if is_best:
            path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, path)
            n_params = sum(v.numel() for v in trainable_state.values())
            size_mb = sum(
                v.numel() * v.element_size() for v in trainable_state.values()
            ) / 1024 / 1024
            logger.info(
                f"保存最佳模型: epoch={epoch}, corr={self.best_val_correlation:.4f}, "
                f"checkpoint={n_params:,} params ({size_mb:.1f}MB)"
            )
