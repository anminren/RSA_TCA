"""DataLoader collate 函数。"""

import torch


def eeg_collate_fn(batch):
    """将 batch 中的样本堆叠为张量。"""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values
    return collated
