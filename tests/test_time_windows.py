import numpy as np

from data.text_eeg_dataset import TextEEGDataset


def _dataset_for_windows():
    dataset = TextEEGDataset.__new__(TextEEGDataset)
    dataset.sampling_rate = 1000
    dataset.baseline_ms = 200
    dataset.time_windows = {"ner": (150, 350), "pos": (0, 500)}
    return dataset


def test_category_specific_windows_and_epoch_fallback():
    dataset = _dataset_for_windows()
    eeg = np.arange(2 * 1201, dtype=np.float32).reshape(2, 1201)

    np.testing.assert_allclose(
        dataset._extract_eeg_target(eeg, "ner"),
        eeg[:, 350:550].mean(axis=1),
    )
    np.testing.assert_allclose(
        dataset._extract_eeg_target(eeg, "noun"),
        eeg[:, 200:700].mean(axis=1),
    )
    np.testing.assert_allclose(
        dataset._extract_eeg_target(eeg, "other"),
        eeg.mean(axis=1),
    )


def test_channel_padding_never_truncates():
    dataset = _dataset_for_windows()
    dataset.max_channels = 4
    np.testing.assert_array_equal(
        dataset._pad_channels(np.array([1.0, 2.0], dtype=np.float32)),
        np.array([1.0, 2.0, 0.0, 0.0], dtype=np.float32),
    )

    with np.testing.assert_raises(ValueError):
        dataset._pad_channels(np.arange(5, dtype=np.float32))
