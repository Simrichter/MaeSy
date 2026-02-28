"""Object detection dataset implementation."""

import os
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np

from maesy.dataset.bounding_box import BoundingBox


class ObjectDetectionDataset(Dataset):
    """Dataset for object detection in COCO format."""
    
    def __init__(
        self,
        dataset_dir: str,
        transforms: Optional[Callable] = None,
        repeat_factor: int = 1,
        step: int = 1,
        start_index: int = 0,
        use_first_n: int = None
    ):
        """
        Initialize ObjectDetectionDataset.
        
        Args:
            :param dataset_dir:
            :param transforms: Optional transforms to apply
        """
        self.images_dir = Path(dataset_dir) / "images"
        self.annotations_dir = Path(dataset_dir) / "labels"
        self.transforms = transforms

        self.images: List[Path] = [Path(img) for img in sorted(os.listdir(self.images_dir)) if img.endswith((".jpg", ".jpeg", ".png"))][start_index::step]*repeat_factor

        self.annotations = []
        for img in self.images:
            annotation_path = Path(img).with_suffix(".txt")
            if not (self.annotations_dir/annotation_path).exists():
                raise FileNotFoundError(f"Annotation file {annotation_path} not found for image {img}")
            self.annotations.append(annotation_path)

        # self.annotations: List[Path] = [Path(ann) for ann in sorted(os.listdir(self.annotations_dir)) if ann.endswith(".txt")][start_index::step]*repeat_factor
        if use_first_n is not None:
            self.images = self.images[:use_first_n]
            self.annotations = self.annotations[:use_first_n]

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[BoundingBox]]:
        """
        Get image and annotations at index.
        If transforms are provided, they are applied to the image before returning it.
        Otherwise, the image is returned as a tensor.
        Labels are returned as a list of BoundingBox instances.
        
        Args:
            :param idx: Index
            
        Returns:
            Dictionary containing the image and target annotations as a List[BoundingBox]
        """
        image_path = os.path.join(self.images_dir, self.images[idx])
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            # Default: convert image to tensor
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        annotation_path = os.path.join(self.annotations_dir, self.annotations[idx])
        if(annotation_path.split("/")[-1].split(".")[0] != image_path.split("/")[-1].split(".")[0]):
            print("\n\nWARNING: Annotation file name does not match image file name! Check that the annotation file names in the labels folder match the image file names in the images folder (except for the extension). Annotation file: {}, Image file: {}\n\n".format(annotation_path, image_path))
        with open(annotation_path, "r") as f:
            boxes = [BoundingBox.from_str(line) for line in f.readlines()]
        return image, boxes
