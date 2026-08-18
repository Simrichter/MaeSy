from typing import List, Dict, Tuple
import torch


def collate_classification_fn(batch: List[tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for classification dataset.

    Args:
        batch: List of (image, label) tuples
    """
    images = torch.stack([image for image, label in batch], dim=0)
    # print(batch[1][1][0]["label"])
    labels = torch.tensor([label[0]["label"] for image, label in batch], dtype=torch.long)
    return images, labels

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



