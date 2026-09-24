import pickle
from pathlib import Path

import pytest

from data.stimulus_loader import StimulusLoader


def test_loads_plain_word_list(tmp_path: Path):
    path = tmp_path / "article_0.pkl"
    path.write_bytes(pickle.dumps(["A", "safe", "sentence", "."]))

    loader = StimulusLoader(str(tmp_path), article_ids=[0])

    assert loader.get_context_for_word(0, 2) == "A safe sentence"
    assert loader.get_target_word(0, 3) == "."


def test_rejects_pickle_globals(tmp_path: Path):
    path = tmp_path / "article_0.pkl"
    path.write_bytes(pickle.dumps(ValueError("must not be constructed")))

    with pytest.raises(pickle.UnpicklingError, match="Refusing non-data pickle global"):
        StimulusLoader(str(tmp_path), article_ids=[0])
