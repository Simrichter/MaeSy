from typing import Any, Optional
import torch


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
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device, non_blocking=True)
        elif isinstance(targets, list):
            if len(targets)>0 and isinstance(targets[0], dict):
                # Move targets to device
                targets_device = []
                for target in targets:
                    td = {k:target[k].to(device, non_blocking=True) for k in target.keys()}
                    targets_device.append(td)
                targets = targets_device
            elif len(targets)>0 and isinstance(targets[0], torch.Tensor):
                targets = [t.to(device, non_blocking=True) for t in targets]

    return images, targets