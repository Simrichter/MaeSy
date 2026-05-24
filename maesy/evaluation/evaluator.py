"""Evaluator for model evaluation."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from ..dataset import MaesyDataset
from ..model import BaseModel, ViTDetector
from .metrics import compute_map, compute_precision_recall, compute_detection_metrics
from maesy.evaluation.inferer import Inferer
from ..model_tools import create_model_from_checkpoint
from torchvision.transforms import v2 as transforms

class Evaluator:
    """Evaluator for Vision Transformer object detection model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        dataset_path: str,
        device: torch.device = ""
    ):
        """
        Initialize evaluator.
        
        Args:
            checkpoint_path: Path to the checkpoint to evaluate
            dataset_path: Path to the MaeSyDataset with a test split to evaluate on
            device: Device to run evaluation on
        """
        if device == "":
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load model
        self.model = create_model_from_checkpoint(checkpoint_path)  # CheckpointHandler(device=device).load_model(checkpoint_path)
        self.model.to(device)
        self.model.eval()

        test_transforms = transforms.Compose(
            [  # TODO: make blank image folder possible again, "auto-infer" split? Maybe through 'None' -> All splits
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize((224, 224)),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        dataset = MaesyDataset(dataset_path, "test", "detection", transforms=test_transforms)
        self.special_classes = dataset.get_special_classes()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
        self.inferer = Inferer(model=self.model, data_loader=dataloader, device=device)


    
    @torch.no_grad()
    def evaluate(
        self,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.
        
        Args:
            confidence_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for matching
            
        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions, all_targets = self.inferer.infer()

        print(compute_detection_metrics(
            predictions=all_predictions,
            targets=all_targets,
            num_classes=self.model.config.num_classes,
            line_class_id=self.special_classes["line_class_id"],
            )
        )

        # Compute metrics
        num_classes = self.model.config.num_classes
        
        map_results = compute_map(
            all_predictions,
            all_targets,
            num_classes,
            iou_threshold
        )
        
        precision, recall = compute_precision_recall(
            all_predictions,
            all_targets,
            iou_threshold
        )
        
        results = {
            'mAP': map_results['mAP'],
            'per_class_AP': map_results['per_class_AP'],
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        }
        
        return results
