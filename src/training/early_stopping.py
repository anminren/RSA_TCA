"""早停机制。"""

import logging

logger = logging.getLogger("eeg_bart")


class ImprovedEarlyStopping:

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "correlation",
        min_epochs: int = 10,
        enabled: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs
        self.enabled = enabled

        self.best_value = -float("inf") if mode == "correlation" else float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, value: float, epoch: int, train_value: float = None) -> bool:
        if not self.enabled:
            return False
        if epoch < self.min_epochs:
            return False

        improved = False
        if self.mode == "correlation":
            if value > self.best_value + self.min_delta:
                improved = True
        else:
            if value < self.best_value - self.min_delta:
                improved = True

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

    def get_status(self) -> str:
        return f"best={self.best_value:.4f}, counter={self.counter}/{self.patience}"
