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

from ..model import BaseModel, ViTDetector
from .metrics import compute_map, compute_precision_recall
from maesy.evaluation.inferer import Inferer


class Evaluator:
    """Evaluator for Vision Transformer object detection model."""
    
    def __init__(
        self,
        model: BaseModel,
        data_loader: Optional[DataLoader] = None,
        device: torch.device = torch.device("cuda")
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader for evaluation (optional for inference-only use)
            device: Device to run evaluation on
        """
        # self.model = model
        # self.data_loader = data_loader
        # self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)

        self.Inferer = Inferer(model, data_loader, device)

    
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
        all_predictions, all_targets = Inferer.infer()
        
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
    
    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        confidence_threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Make prediction on a single image.
        
        Args:
            image: Input image tensor [C, H, W]
            confidence_threshold: Confidence threshold
            
        Returns:
            Dictionary with predictions
        """
        self.model.eval()
        
        # Add batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Get predictions
        predictions = self.model.inference(image, confidence_threshold)
        
        return predictions[0]
    
    def visualize_predictions(
        self,
        image: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
        category_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize predictions on image.
        
        Args:
            image: Input image tensor [C, H, W]
            predictions: Predictions dictionary
            category_names: List of category names
            save_path: Path to save visualization
        """
        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            if image.shape[0] == 3:  # [C, H, W]
                image = np.transpose(image, (1, 2, 0))
        
        # Denormalize if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw boxes
        boxes = predictions['boxes'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        
        h, w = image.shape[:2]
        
        for box, label, score in zip(boxes, labels, scores):
            # Model outputs boxes in [cx, cy, w, h] normalized format
            # Convert to [x1, y1, x2, y2] pixel coordinates
            cx, cy, bw, bh = box
            cx, cy = cx * w, cy * h
            bw, bh = bw * w, bh * h
            
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            
            # Draw box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label_text = f"{category_names[label] if category_names else label}: {score:.2f}"
            ax.text(
                x1,
                y1 - 5,
                label_text,
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=10,
                color='white'
            )
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def evaluate_model(
    model: ViTDetector,
    data_loader: DataLoader,
    device: str = "cuda",
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Convenience function for model evaluation.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use
        confidence_threshold: Confidence threshold
        iou_threshold: IoU threshold
        
    Returns:
        Evaluation results
    """
    evaluator = Evaluator(model, data_loader, device)
    results = evaluator.evaluate(confidence_threshold, iou_threshold)
    
    print("\n=== Evaluation Results ===")
    print(f"mAP@{iou_threshold}: {results['mAP']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    return results
