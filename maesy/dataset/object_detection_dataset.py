"""Object detection dataset implementation."""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Dict, Any, List
import numpy as np


class ObjectDetectionDataset(Dataset):
    """Dataset for object detection in COCO format."""
    
    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        transforms: Optional[Callable] = None,
        num_classes: Optional[int] = None
    ):
        """
        Initialize ObjectDetectionDataset.
        
        Args:
            images_dir: Directory containing images
            annotations_file: Path to COCO format annotations JSON
            transforms: Optional transforms to apply
            num_classes: Number of object classes (inferred if None)
        """
        self.images_dir = images_dir
        self.annotations_file = annotations_file
        self.transforms = transforms
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data.get('images', [])
        self.annotations = self.coco_data.get('annotations', [])
        self.categories = self.coco_data.get('categories', [])
        
        # Create image id to annotations mapping
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)
        
        # Create category id to index mapping
        self.cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(self.categories)}
        
        if num_classes is None:
            self.num_classes = len(self.categories)
        else:
            self.num_classes = num_classes
            
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get image and annotations at index.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing image and target annotations
        """
        image_info = self.images[idx]
        image_id = image_info['id']
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Get annotations for this image
        annotations = self.image_id_to_annotations.get(image_id, [])
        
        # Convert to target format
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[ann['category_id']])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target['image_id'] = torch.tensor([image_id])
        target['area'] = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            # Default: convert image to tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        return {'image': image, 'target': target}
    
    def get_category_names(self) -> List[str]:
        """Get list of category names."""
        return [cat['name'] for cat in self.categories]
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.num_classes
