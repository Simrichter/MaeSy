from typing import Any, Optional

import torch
from torch.utils.data import Dataset


def handle_raw_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract images and targets from raw batch data.

    :param batch: Raw batch data (dict, list/tuple, or tensor)
    :param device: The device to move the tensors to
    :return: Tuple of images and targets (targets can be None)
    """
    targets = None
    if isinstance(batch, dict):
        images = batch['images']
        targets = batch['targets']
    elif isinstance(batch, (list, tuple)):
        images = batch[0]
        targets = batch[1]
    else:
        images = batch

    images = images.to(device, non_blocking=True)
    if targets is not None:
        targets = batch['targets'].to(device, non_blocking=True)
    return images, targets



