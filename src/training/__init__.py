"""训练模块。"""

from .trainer import EEGBARTTrainer
from .early_stopping import ImprovedEarlyStopping

__all__ = ["EEGBARTTrainer", "ImprovedEarlyStopping"]
