"""Object detection dataset implementation."""

import os
import json
from pathlib import Path

import torch
import torchvision.tv_tensors
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np
from torchvision.ops import box_convert

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
    
    def __getitem__(self, idx: int) -> Tuple[torchvision.tv_tensors.Image, List[Dict[str, torchvision.tv_tensors.BoundingBoxes]]]:
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
        with Image.open(image_path).convert('RGB') as image:
            img_width, img_height = image.size
            annotation_path = os.path.join(self.annotations_dir, self.annotations[idx])
            if annotation_path.split("/")[-1].split(".")[0] != image_path.split("/")[-1].split(".")[0]:
                print(
                    "\n\nWARNING: Annotation file name does not match image file name! Check that the annotation file names in the labels folder match the image file names in the images folder (except for the extension). Annotation file: {}, Image file: {}\n\n".format(
                        annotation_path, image_path))
            with open(annotation_path, "r") as f:
                boxes_list = [BoundingBox.from_str(line) for line in f.readlines()]
                for box in boxes_list:
                    box.scale_to_size(img_width, img_height) # TODO: Ugly
                if len(boxes_list) > 0:
                    coords_np = np.array([box.as_cxcywh() for box in boxes_list], dtype=np.float32)
                    coords = torch.from_numpy(coords_np)

                    # Wrap as tv_tensors.BoundingBoxes with actual image size
                    coords = torchvision.tv_tensors.BoundingBoxes(
                        coords,
                        format="CXCYWH",
                        canvas_size=(img_height, img_width)  # (H, W)
                    )
                    labels = torch.tensor([box.cls_id for box in boxes_list], dtype=torch.long)
                else:
                    coords = torchvision.tv_tensors.BoundingBoxes(
                        torch.empty((0, 4), dtype=torch.float32),
                        format="CXCYWH",
                        canvas_size=(img_height, img_width)
                    )
                    labels = torch.empty((0,), dtype=torch.long)

                target = {"boxes": coords, "labels": labels}
            image = torchvision.tv_tensors.Image(image)

            if self.transforms is not None:
                image, target = self.transforms(image, target)
                # Filter out invalid boxes (out of bounds or zero area)
                boxes = target["boxes"]
                labels = target["labels"]

                # Convert to xyxy to check validity
                boxes_xyxy = box_convert(boxes, "cxcywh", "xyxy")

                # Keep boxes that have area > 0
                valid_mask = (boxes_xyxy[:, 2]-boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) > 0

                target["boxes"] = boxes[valid_mask]
                target["labels"] = labels[valid_mask]

                # Normalize boxes back to [0,1]
                h, w = image.shape[-2:]
                if len(target["boxes"]) > 0:
                    target["boxes"] = target["boxes"] / torch.tensor([w, h, w, h], device=target["boxes"].device)
            else:
                # Default: convert image to tensor
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                # Normalize boxes back to [0,1] after transforms
                h, w = image.shape[-2:]
                target["boxes"] = target["boxes"] / torch.tensor([w, h, w, h], device=target["boxes"].device)

            return image, target
