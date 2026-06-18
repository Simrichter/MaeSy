from abc import ABC, abstractmethod
from typing import Optional

import torch
import numpy as np
from markdown_it.rules_inline import newline
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from maesy.dataset import UnlabeledDataset, MultiDataset, MaesyDataset
from maesy.dataset.od_augmentations import ClusterTransforms
from maesy.evaluation.inferer import Inferer
from maesy.model import ResnetFeatureExtractor, BaseModel

class BaseClustering(ABC):

    def run(self, paths: list[str], chosen_paths: Optional[list[str]]=None, batch_size=256, forward_scale=224, step=1, start_index=0, **kwargs):
        """
        Execute the specific clustering
        """
        # Setup transforms for feature extraction
        # img_transforms = transforms.Compose([
        #     transforms.Resize(size=(forward_scale, forward_scale)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard ImageNet normalization
        # ])
        img_transforms = ClusterTransforms(image_size=forward_scale)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(paths) > 0:
            # Create dataset from all image directories
            internal_dataset = MultiDataset(
                [MaesyDataset(dataset_dir=path, annotation_type="image_folder", transforms=img_transforms, step=step, start_index=start_index) for path in
                 paths])
        else:
            raise ValueError("Failed: No data path specified!")

        new_dataloader = DataLoader(internal_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, in_order=True)

        existing_dataloader = None
        if chosen_paths is not None:
            print(f"Adding pre-chosen images to FAISS index...")
            # Create dataset from all image directories
            chosen_multi = MultiDataset([
                MaesyDataset(dataset_dir=path, annotation_type="image_folder", transforms=img_transforms, step=step, start_index=start_index) for path in
                chosen_paths
            ])
            existing_dataloader = DataLoader(chosen_multi, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)


        model = self._create_model(forward_scale)

        return self._cluster(new_dataloader, existing_dataloader, model, **kwargs)

    @abstractmethod
    def _create_model(self, forward_scale):
        pass

    @abstractmethod
    def _cluster(self, new_dataloader: DataLoader, existing_dataloader: Optional[DataLoader], model: BaseModel, **kwargs):
        pass

