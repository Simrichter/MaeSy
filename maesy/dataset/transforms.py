"""Data transforms for training and validation."""

import torch
import torchvision.transforms as T
from typing import Tuple, Dict, Any
import random


def get_train_transforms(image_size: int = 224):
    """
    Get training transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Callable transform function
    """
    def transform(image, target):
        # Convert PIL Image to tensor
        image = T.functional.to_tensor(image)
        
        # Normalize
        image = T.functional.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Resize
        orig_h, orig_w = image.shape[1], image.shape[2]
        image = T.functional.resize(image, [image_size, image_size])
        
        # Scale boxes accordingly
        if 'boxes' in target and target['boxes'].numel() > 0:
            scale_x = image_size / orig_w
            scale_y = image_size / orig_h
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target['boxes'] = boxes
        
        return image, target
    
    return transform


def get_val_transforms(image_size: int = 224):
    """
    Get validation transforms.
    
    Args:
        image_size: Target image size
        
    Returns:
        Callable transform function
    """
    def transform(image, target):
        # Convert PIL Image to tensor
        image = T.functional.to_tensor(image)
        
        # Normalize
        image = T.functional.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Resize
        orig_h, orig_w = image.shape[1], image.shape[2]
        image = T.functional.resize(image, [image_size, image_size])
        
        # Scale boxes accordingly
        if 'boxes' in target and target['boxes'].numel() > 0:
            scale_x = image_size / orig_w
            scale_y = image_size / orig_h
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            target['boxes'] = boxes
        
        return image, target
    
    return transform


def collate_fn(batch):
    """Custom collate function for batching."""
    images = []
    targets = []
    
    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
    
    images = torch.stack(images, 0)
    
    return {'images': images, 'targets': targets}
