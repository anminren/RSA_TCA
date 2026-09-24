"""损失函数模块。"""

from .brain_loss import BrainLoss
from .moco_loss import MoCoLoss
from .anchor_loss import AnchorLoss
from .cosine_anchor_loss import CosineAnchorLoss
from .kl_anchor_loss import KLAnchorLoss
from .combined_loss import CombinedLoss

__all__ = [
    "BrainLoss",
    "MoCoLoss",
    "AnchorLoss",
    "CosineAnchorLoss",
    "KLAnchorLoss",
    "CombinedLoss",
]
