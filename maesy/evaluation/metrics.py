"""Evaluation metrics for object detection."""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes.
    
    Args:
        box1: Box in [x1, y1, x2, y2] format
        box2: Box in [x1, y1, x2, y2] format
        
    Returns:
        IoU value
    """
    # Compute intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Compute union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_precision_recall(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Compute precision and recall.
    
    Args:
        predictions: List of predictions, each with 'boxes', 'labels', 'scores'
        targets: List of ground truth targets
        iou_threshold: IoU threshold for matching
        
    Returns:
        Tuple of (precision, recall)
    """
    total_tp = 0
    total_fp = 0
    total_gt = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        target_boxes = target['boxes'].cpu().numpy()
        target_labels = target['labels'].cpu().numpy()
        
        total_gt += len(target_boxes)
        
        # Track which targets have been matched
        matched_targets = set()
        
        # Sort predictions by score (descending)
        sorted_idx = np.argsort(-pred_scores)
        
        for idx in sorted_idx:
            pred_box = pred_boxes[idx]
            pred_label = pred_labels[idx]
            
            best_iou = 0.0
            best_target_idx = -1
            
            # Find best matching target
            for target_idx, (target_box, target_label) in enumerate(zip(target_boxes, target_labels)):
                if target_idx in matched_targets:
                    continue
                
                if pred_label != target_label:
                    continue
                
                iou = compute_iou(pred_box, target_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx
            
            # Check if match is valid
            if best_iou >= iou_threshold:
                total_tp += 1
                matched_targets.add(best_target_idx)
            else:
                total_fp += 1
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    
    return precision, recall


def compute_ap(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    class_id: int,
    iou_threshold: float = 0.5
) -> float:
    """
    Compute Average Precision for a single class.
    
    Args:
        predictions: List of predictions
        targets: List of ground truth targets
        class_id: Class ID to compute AP for
        iou_threshold: IoU threshold
        
    Returns:
        Average Precision value
    """
    # Collect all predictions and targets for this class
    all_pred_boxes = []
    all_pred_scores = []
    all_target_boxes = []
    
    for pred, target in zip(predictions, targets):
        # Filter predictions for this class
        pred_mask = pred['labels'].cpu().numpy() == class_id
        if pred_mask.any():
            pred_boxes = pred['boxes'].cpu().numpy()[pred_mask]
            pred_scores = pred['scores'].cpu().numpy()[pred_mask]
            
            for box, score in zip(pred_boxes, pred_scores):
                all_pred_boxes.append(box)
                all_pred_scores.append(score)
        
        # Filter targets for this class
        target_mask = target['labels'].cpu().numpy() == class_id
        if target_mask.any():
            target_boxes = target['boxes'].cpu().numpy()[target_mask]
            all_target_boxes.extend(target_boxes)
    
    if len(all_pred_boxes) == 0 or len(all_target_boxes) == 0:
        return 0.0
    
    # Sort predictions by score
    sorted_idx = np.argsort(-np.array(all_pred_scores))
    all_pred_boxes = [all_pred_boxes[i] for i in sorted_idx]
    all_pred_scores = [all_pred_scores[i] for i in sorted_idx]
    
    # Compute precision-recall curve
    tp = np.zeros(len(all_pred_boxes))
    fp = np.zeros(len(all_pred_boxes))
    matched_targets = set()
    
    for pred_idx, (pred_box, pred_score) in enumerate(zip(all_pred_boxes, all_pred_scores)):
        best_iou = 0.0
        best_target_idx = -1
        
        for target_idx, target_box in enumerate(all_target_boxes):
            if target_idx in matched_targets:
                continue
            
            iou = compute_iou(pred_box, target_box)
            
            if iou > best_iou:
                best_iou = iou
                best_target_idx = target_idx
        
        if best_iou >= iou_threshold:
            tp[pred_idx] = 1
            matched_targets.add(best_target_idx)
        else:
            fp[pred_idx] = 1
    
    # Compute cumulative tp and fp
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute precision and recall
    recalls = tp_cumsum / len(all_target_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap


def compute_map(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP).
    
    Args:
        predictions: List of predictions
        targets: List of ground truth targets
        num_classes: Number of classes
        iou_threshold: IoU threshold
        
    Returns:
        Dictionary with mAP and per-class AP
    """
    aps = []
    
    for class_id in range(num_classes):
        ap = compute_ap(predictions, targets, class_id, iou_threshold)
        aps.append(ap)
    
    results = {
        'mAP': np.mean(aps),
        'per_class_AP': aps
    }
    
    return results
