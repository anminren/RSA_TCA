"""Shared tokenization utilities.

Supports fast tokenizers with offset mappings and slow SentencePiece tokenizers
(such as Pegasus) with a token-id fallback for locating the target word.
"""

from typing import List

import torch
from transformers import PreTrainedTokenizer


def _pad_or_left_truncate(
    full_input_ids: List[int],
    full_attention_mask: List[int],
    target_token_indices: List[int],
    pad_token_id: int,
    max_length: int,
) -> tuple:
    full_length = len(full_input_ids)
    if full_length <= max_length:
        pad_length = max_length - full_length
        input_ids = full_input_ids + [pad_token_id] * pad_length
        attention_mask = full_attention_mask + [0] * pad_length
        target_word_mask = [0] * max_length
        for tok_idx in target_token_indices:
            if 0 <= tok_idx < max_length:
                target_word_mask[tok_idx] = 1
    else:
        truncate_offset = full_length - max_length
        input_ids = full_input_ids[truncate_offset:]
        attention_mask = full_attention_mask[truncate_offset:]
        target_word_mask = [0] * max_length
        for tok_idx in target_token_indices:
            new_idx = tok_idx - truncate_offset
            if 0 <= new_idx < max_length:
                target_word_mask[new_idx] = 1
    return input_ids, attention_mask, target_word_mask


def _find_subsequence_last(sequence: List[int], pattern: List[int]) -> List[int]:
    if not pattern:
        return []
    matches = []
    n = len(pattern)
    for i in range(0, len(sequence) - n + 1):
        if sequence[i : i + n] == pattern:
            matches = list(range(i, i + n))
    return matches


def _slow_target_indices(tokenizer, context_text: str, target_word: str, input_ids: List[int]) -> List[int]:
    candidates = []
    for text in (target_word, " " + target_word):
        ids = tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
        if ids and ids not in candidates:
            candidates.append(ids)

    target_start_char = context_text.rfind(target_word)
    if target_start_char >= 0:
        suffix = context_text[target_start_char:]
        ids = tokenizer(suffix, add_special_tokens=False, return_tensors=None)["input_ids"]
        if ids and ids not in candidates:
            candidates.append(ids)

    for ids in candidates:
        found = _find_subsequence_last(input_ids, ids)
        if found:
            return found

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    non_special = [i for i, tok_id in enumerate(input_ids) if tok_id not in special_ids]
    return [non_special[-1]] if non_special else []


def tokenize_with_left_truncation(
    tokenizer: PreTrainedTokenizer,
    context_text: str,
    target_word: str,
    max_length: int,
) -> tuple:
    """Tokenize context text with left truncation and target-word masking."""
    if not target_word:
        raise ValueError("target_word must be non-empty")
    context_words = context_text.split()
    if len(context_words) > max_length * 10:
        context_text = " ".join(context_words[-(max_length * 10):])

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    if getattr(tokenizer, "is_fast", False):
        encoding = tokenizer(
            context_text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            return_tensors=None,
        )
        full_input_ids = encoding["input_ids"]
        full_attention_mask = encoding["attention_mask"]
        offset_mapping = encoding["offset_mapping"]

        target_start_char = context_text.rfind(target_word)
        if target_start_char < 0:
            raise ValueError(f"Target word {target_word!r} is absent from its context")
        target_end_char = target_start_char + len(target_word)
        target_token_indices = []
        for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start == 0 and char_end == 0:
                continue
            if char_start < target_end_char and char_end > target_start_char:
                target_token_indices.append(tok_idx)
    else:
        encoding = tokenizer(
            context_text,
            add_special_tokens=True,
            return_tensors=None,
        )
        full_input_ids = encoding["input_ids"]
        full_attention_mask = encoding["attention_mask"]
        target_token_indices = _slow_target_indices(
            tokenizer, context_text, target_word, full_input_ids
        )

    input_ids, attention_mask, target_word_mask = _pad_or_left_truncate(
        full_input_ids,
        full_attention_mask,
        target_token_indices,
        pad_token_id,
        max_length,
    )

    if not any(target_word_mask):
        raise ValueError(
            f"Could not align target word {target_word!r} after tokenization/truncation"
        )

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(target_word_mask, dtype=torch.long),
    )
