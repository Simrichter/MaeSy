from typing import Any, Optional, List, Dict, Tuple

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
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device, non_blocking=True)
        elif isinstance(targets, list):
            if len(targets)>0 and isinstance(targets[0], dict):
                # Move targets to device
                targets_device = []
                for target in targets:
                    targets_device.append({
                        'boxes': target['boxes'].to(device, non_blocking=True),
                        'labels': target['labels'].to(device, non_blocking=True)
                    })
                targets = targets_device
            elif len(targets)>0 and isinstance(targets[0], torch.Tensor):
                targets = [t.to(device, non_blocking=True) for t in targets]

    return images, targets


def collate_detection_fn(batch: List[tuple]) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for object detection dataset.
    
    Args:
        batch: List of (image, targets) tuples where targets is a list of BoundingBox
        
    Returns:
        Tuple of:
            - images: Stacked images tensor [B, C, H, W]
            - targets: List of target dictionaries with 'boxes' [N X  and 'labels'
    """
    images = torch.stack([image for image, boxes in batch], dim=0)
    targets = [boxes for image, boxes in batch]
    # for ba in batch:
    #     image, boxes = ba[0], ba[1]
    #     images.append(image)

        # if len(boxes) > 0:
        #     # Convert BoundingBox objects to tensors
        #     labels = boxes["labels"]
        #     # Get normalized coordinates in [cx, cy, w, h] format
        #     boxes_tensor = torch.tensor([box.as_cxcywh() for box in boxes], dtype=torch.float32)
        #
        #     targets.append({
        #         'labels': labels,
        #         'boxes': boxes_tensor
        #     })
        # else:
        #     # Handle images with no boxes
        #     targets.append({
        #         'labels': torch.tensor([], dtype=torch.long),
        #         'boxes': torch.tensor([], dtype=torch.float32).reshape(0, 4)
        #     })

    # Stack images
    # images = torch.stack(images, dim=0)
    
    return images, targets



