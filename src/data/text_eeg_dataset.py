"""文本-EEG 配对数据集（支持 Derco 和 Frank 两种数据集格式）。

支持 Derco 和 Frank 两种数据集格式：
- Derco: {subject_id}/article_N/preprocessed_epoch.npy + event_info.json, 32ch, 1000Hz
- Frank: art_N/Overall/sub_X_art_N_epo.npy, 28ch, 250Hz
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .tokenize_utils import tokenize_with_left_truncation

logger = logging.getLogger("eeg_bart")


def _ms_to_samples(time_ms: int, sampling_rate: int, baseline_ms: int) -> int:
    return int((baseline_ms + time_ms) * sampling_rate / 1000)


class TextEEGDataset(Dataset):
    """文本上下文 + EEG 脑信号配对数据集。"""

    def __init__(
        self,
        stimulus_loader,
        tokenizer: PreTrainedTokenizer,
        subject_id: str,
        article_ids: List[int],
        eeg_data_dir: str,
        max_length: int = 128,
        target_type: str = "average",
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
        split_seed: int = 42,
        permute_eeg: bool = False,
        permute_seed: int = 42,
        dataset_type: str = "derco",
        sampling_rate: int = 1000,
        baseline_ms: int = 200,
        max_channels: int = 32,
        time_windows: Optional[Dict[str, Tuple[int, int]]] = None,
        word_category_cache=None,
        n_folds: Optional[int] = None,
        current_fold: Optional[int] = None,
        max_timepoints: Optional[int] = None,
    ):
        self.stimulus_loader = stimulus_loader
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_type = target_type
        self.permute_eeg = permute_eeg
        self.dataset_type = dataset_type
        self.sampling_rate = sampling_rate
        self.baseline_ms = baseline_ms
        self.max_channels = max_channels
        self.time_windows = time_windows
        self.word_category_cache = word_category_cache
        self.max_timepoints = max_timepoints

        self.samples: List[Dict] = []
        if dataset_type == "derco":
            self._load_derco(subject_id, article_ids, eeg_data_dir)
        elif dataset_type == "frank":
            self._load_frank(subject_id, article_ids, eeg_data_dir)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        if n_folds is not None and current_fold is not None:
            self._apply_kfold_split(split, n_folds, current_fold)
        else:
            self._apply_split(split, split_ratio, split_seed)

        if permute_eeg:
            rng = np.random.RandomState(permute_seed)
            eeg_indices = rng.permutation(len(self.samples)).tolist()
            eeg_data = [self.samples[i]["eeg_data"] for i in eeg_indices]
            for i, sample in enumerate(self.samples):
                sample["eeg_data"] = eeg_data[i]

    def _load_derco(
        self, subject_id: str, article_ids: List[int], eeg_data_dir: str
    ) -> None:
        data_dir = Path(eeg_data_dir)

        for article_id in article_ids:
            article_dir = data_dir / subject_id / f"article_{article_id}"
            npy_path = article_dir / "preprocessed_epoch.npy"
            json_path = article_dir / "event_info.json"

            if not npy_path.exists():
                logger.warning("DERCo EEG not found: %s", npy_path)
                continue
            if not json_path.exists():
                raise FileNotFoundError(f"DERCo event metadata not found: {json_path}")

            eeg_data = np.load(npy_path, allow_pickle=False)
            with open(json_path, "r") as f:
                event_info = json.load(f)

            event_numbers = event_info["event_numbers"]
            if eeg_data.ndim != 3:
                raise ValueError(
                    f"Expected DERCo EEG shape [epochs, channels, time], got "
                    f"{eeg_data.shape} in {npy_path}"
                )
            if len(event_numbers) != eeg_data.shape[0]:
                raise ValueError(
                    f"event_numbers ({len(event_numbers)}) != epochs "
                    f"({eeg_data.shape[0]}) in {article_dir}"
                )
            article_length = self.stimulus_loader.get_article_length(article_id)

            if self.max_timepoints is not None:
                eeg_data = eeg_data[..., : self.max_timepoints]

            for epoch_idx, word_position in enumerate(event_numbers):
                if not isinstance(word_position, int) or not 0 <= word_position < article_length:
                    raise ValueError(
                        f"Invalid event number {word_position!r} for article_{article_id} "
                        f"with {article_length} words"
                    )
                self.samples.append({
                    "article_id": article_id,
                    "word_position": word_position,
                    "epoch_idx": epoch_idx,
                    "eeg_data": eeg_data[epoch_idx],
                })

    def _load_frank(
        self, subject_id: str, article_ids: List[int], eeg_data_dir: str
    ) -> None:
        data_dir = Path(eeg_data_dir)

        for article_id in article_ids:
            npy_path = (
                data_dir / f"art_{article_id}" / "Overall"
                / f"{subject_id}_art_{article_id}_epo.npy"
            )

            if not npy_path.exists():
                logger.warning(f"Frank EEG not found: {npy_path}")
                continue

            eeg_data = np.load(npy_path, allow_pickle=False)
            if eeg_data.ndim != 3:
                raise ValueError(
                    f"Expected FRANK EEG shape [epochs, channels, time], got "
                    f"{eeg_data.shape} in {npy_path}"
                )
            n_epochs = eeg_data.shape[0]
            n_words = self.stimulus_loader.get_article_length(article_id)
            n_valid = min(n_epochs, n_words)

            if n_epochs != n_words:
                logger.warning(
                    f"Frank art_{article_id} {subject_id}: "
                    f"epochs={n_epochs} != words={n_words}, using first {n_valid}"
                )

            for word_position in range(n_valid):
                self.samples.append({
                    "article_id": article_id,
                    "word_position": word_position,
                    "epoch_idx": word_position,
                    "eeg_data": eeg_data[word_position],
                })

    def _apply_split(
        self, split: str, split_ratio: tuple, seed: int
    ) -> None:
        rng = np.random.RandomState(seed)
        n = len(self.samples)
        indices = rng.permutation(n)

        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train:n_train + n_val]
        elif split == "test":
            selected = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.samples = [self.samples[i] for i in sorted(selected)]

    def _apply_kfold_split(
        self, split: str, n_folds: int, current_fold: int
    ) -> None:
        n = len(self.samples)
        indices = np.arange(n)

        fold_sizes = [n // n_folds] * n_folds
        for i in range(n % n_folds):
            fold_sizes[i] += 1

        fold_start = sum(fold_sizes[:current_fold])
        fold_end = fold_start + fold_sizes[current_fold]
        val_indices = indices[fold_start:fold_end]

        if split == "train":
            selected = np.concatenate([indices[:fold_start], indices[fold_end:]])
        elif split == "val":
            selected = val_indices
        else:
            raise ValueError(f"K-fold mode only supports 'train'/'val', got: {split}")

        self.samples = [self.samples[i] for i in selected]

    def _extract_eeg_target(self, eeg: np.ndarray, category: str) -> np.ndarray:
        """根据词汇类别提取对应时间窗口的 EEG 特征。"""
        window = None
        if self.time_windows:
            if category == "ner" and "ner" in self.time_windows:
                window = self.time_windows["ner"]
            elif category in ("noun", "verb") and "pos" in self.time_windows:
                window = self.time_windows["pos"]

        if window is not None:
            start_ms, end_ms = window
            start_sample = _ms_to_samples(start_ms, self.sampling_rate, self.baseline_ms)
            end_sample = _ms_to_samples(end_ms, self.sampling_rate, self.baseline_ms)
            start_sample = max(0, min(start_sample, eeg.shape[1] - 1))
            end_sample = max(start_sample + 1, min(end_sample, eeg.shape[1]))
            return eeg[:, start_sample:end_sample].mean(axis=1)

        return eeg.mean(axis=1)

    def _pad_channels(self, eeg_target: np.ndarray) -> np.ndarray:
        """将 EEG 特征零填充到 max_channels 维度。"""
        n_ch = eeg_target.shape[0]
        if n_ch < self.max_channels:
            padded = np.zeros(self.max_channels, dtype=eeg_target.dtype)
            padded[:n_ch] = eeg_target
            return padded
        if n_ch > self.max_channels:
            raise ValueError(
                f"EEG has {n_ch} channels but max_channels={self.max_channels}; "
                "refusing to silently discard channels"
            )
        return eeg_target

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        article_id = sample["article_id"]
        word_position = sample["word_position"]

        context_text = self.stimulus_loader.get_context_for_word(
            article_id, word_position
        )
        target_word = self.stimulus_loader.get_target_word(
            article_id, word_position
        )

        input_ids, attention_mask, target_word_mask = tokenize_with_left_truncation(
            self.tokenizer, context_text, target_word, self.max_length
        )

        eeg = sample["eeg_data"]

        category = "other"
        if self.word_category_cache is not None:
            article_key = f"{self.dataset_type}_{article_id}"
            words = self.stimulus_loader.get_article_words(article_id)
            category = self.word_category_cache.get_word_category(
                article_key, words, word_position
            )

        if self.target_type == "average":
            eeg_target = self._extract_eeg_target(eeg, category)
        else:
            eeg_target = eeg.mean(axis=1)

        eeg_target = self._pad_channels(eeg_target)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_word_mask": target_word_mask,
            "eeg_target": torch.from_numpy(eeg_target).float(),
            "word_idx": torch.tensor(word_position, dtype=torch.long),
            "word_category": category,
        }
