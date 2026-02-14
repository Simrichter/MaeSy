from typing import Any, Optional, List, Dict

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


def collate_detection_fn(batch: List[tuple]) -> tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
    """
    Custom collate function for object detection dataset.
    
    Args:
        batch: List of (image, targets) tuples where targets is a list of BoundingBox
        
    Returns:
        Tuple of:
            - images: Stacked images tensor [B, C, H, W]
            - targets: List of target dictionaries with 'boxes' and 'labels'
    """
    from maesy.dataset.bounding_box import BoundingBox
    
    images = []
    targets = []
    
    for image, boxes in batch:
        images.append(image)
        
        if len(boxes) > 0:
            # Convert BoundingBox objects to tensors
            labels = torch.tensor([box.cls_id for box in boxes], dtype=torch.long)
            # Get normalized coordinates in [cx, cy, w, h] format
            boxes_tensor = torch.tensor([box.as_xywh() for box in boxes], dtype=torch.float32)
            
            targets.append({
                'labels': labels,
                'boxes': boxes_tensor
            })
        else:
            # Handle images with no boxes
            targets.append({
                'labels': torch.tensor([], dtype=torch.long),
                'boxes': torch.tensor([], dtype=torch.float32).reshape(0, 4)
            })
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    return images, targets



