"""Object detection dataset implementation."""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Dict, Any
import numpy as np


class UnlabeledDataset(Dataset):
    """Dataset for unlabeled images."""
    
    def __init__(
        self,
        images_dir: str | Path,
        filetype: str = ".png",
        transforms: Optional[Callable] = None,
        repeat_factor: int = 1,
        use_first_n: int = None
    ):
        """
        Initialize UnlabeledDataset.
        
        Args:
            images_dir: Directory containing images
            filetype: File extension
            transforms: Optional transforms to apply
            repeat_factor: Specifies, how many times the same image is repeated in the dataset (mostly for debugging)
            use_first_n: Only use the first n images found in the images_dir (mostly for debugging)
        """
        self.images_dir = images_dir
        self.transforms = transforms

        self.images = [f for f in sorted(os.listdir(self.images_dir)) if f.endswith(filetype)]*repeat_factor
        if use_first_n is not None:
            self.images = self.images[:use_first_n]

        if len(self.images) == 0:
            print(f"Warning: No images of type {filetype} found in this directory: {self.images_dir}")
            
    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get image and annotations at index.
        
        Args:
            idx: Index
            
        Returns:
            The image as a tensor
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
            
        return image