import pytest

from data.tokenize_utils import tokenize_with_left_truncation


class FakeFastTokenizer:
    is_fast = True
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        words = text.split()
        ids = [101]
        offsets = [(0, 0)]
        cursor = 0
        for index, word in enumerate(words, start=1):
            start = text.find(word, cursor)
            end = start + len(word)
            ids.append(index)
            offsets.append((start, end))
            cursor = end
        ids.append(102)
        offsets.append((0, 0))
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": offsets,
        }


def test_target_word_mask_tracks_last_word():
    _, _, mask = tokenize_with_left_truncation(
        FakeFastTokenizer(), "the final word", "word", max_length=8
    )
    assert mask.tolist() == [0, 0, 0, 1, 0, 0, 0, 0]


def test_missing_target_is_rejected():
    with pytest.raises(ValueError, match="absent"):
        tokenize_with_left_truncation(
            FakeFastTokenizer(), "the final word", "missing", max_length=8
        )
