"""Evaluation metrics for object detection."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torchvision.ops import box_convert


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in ``xyxy`` format."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - intersection
    return intersection / union if union > 0.0 else 0.0


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)
    return box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")


def _sanitize_xyxy(boxes_xyxy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.reshape(0, 4), torch.zeros((0,), dtype=torch.bool)
    boxes_xyxy = boxes_xyxy.clone()
    boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(0.0, 1.0)
    boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(0.0, 1.0)
    valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
    return boxes_xyxy[valid], valid


def decode_detr_predictions(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    no_object_class: int | None = None,
    score_threshold: float = 0.0,
) -> List[Dict[str, torch.Tensor]]:
    """Decode DETR outputs to a list of ``xyxy`` detections per image."""
    if no_object_class is None:
        no_object_class = pred_logits.shape[-1] - 1

    probs = pred_logits.softmax(-1)
    scores, labels = probs.max(-1)
    decoded: List[Dict[str, torch.Tensor]] = []

    for img_idx in range(pred_logits.shape[0]):
        mask = labels[img_idx] != no_object_class
        if score_threshold > 0.0:
            mask = mask & (scores[img_idx] >= score_threshold)

        boxes = pred_boxes[img_idx][mask].detach().cpu().float()
        boxes_xyxy, valid = _sanitize_xyxy(_cxcywh_to_xyxy(boxes))

        # Sanitize labels/scores with same valid mask used for boxes.
        if boxes.numel() == 0:
            det_labels = torch.empty((0,), dtype=torch.long)
            det_scores = torch.empty((0,), dtype=torch.float32)
        else:
            initial_labels = labels[img_idx][mask].detach().cpu().long()
            initial_scores = scores[img_idx][mask].detach().cpu().float()
            det_labels = initial_labels[valid]
            det_scores = initial_scores[valid]

        decoded.append({
            "boxes": boxes_xyxy,
            "labels": det_labels,
            "scores": det_scores,
        })

    return decoded


def prepare_targets_for_detection_metrics(targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """Convert training targets from normalized ``cxcywh`` to ``xyxy``."""
    prepared: List[Dict[str, torch.Tensor]] = []
    for target in targets:
        boxes = target["boxes"].detach().cpu().float()
        labels = target["labels"].detach().cpu().long()
        boxes_xyxy, valid = _sanitize_xyxy(_cxcywh_to_xyxy(boxes))
        if boxes.numel() == 0:
            labels = torch.empty((0,), dtype=torch.long)
        else:
            labels = labels[valid]
        prepared.append({"boxes": boxes_xyxy, "labels": labels})
    return prepared


def _iter_detections_for_class(
    predictions: List[Dict[str, torch.Tensor]],
    class_id: int,
) -> List[Tuple[int, float, np.ndarray]]:
    detections: List[Tuple[int, float, np.ndarray]] = []
    for image_idx, pred in enumerate(predictions):
        if pred["boxes"].numel() == 0:
            continue
        class_mask = pred["labels"] == class_id
        boxes = pred["boxes"][class_mask]
        scores = pred["scores"][class_mask]
        for box, score in zip(boxes.numpy(), scores.numpy()):
            detections.append((image_idx, float(score), box))
    detections.sort(key=lambda item: item[1], reverse=True)
    return detections


def _ground_truth_by_image(
    targets: List[Dict[str, torch.Tensor]],
    class_id: int,
) -> Dict[int, np.ndarray]:
    ground_truth: Dict[int, np.ndarray] = {}
    for image_idx, target in enumerate(targets):
        if target["boxes"].numel() == 0:
            ground_truth[image_idx] = np.zeros((0, 4), dtype=np.float32)
            continue
        class_mask = target["labels"] == class_id
        ground_truth[image_idx] = target["boxes"][class_mask].numpy().astype(np.float32)
    return ground_truth


def _compute_pr_curve(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    class_id: int,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    detections = _iter_detections_for_class(predictions, class_id)
    gt_by_img = _ground_truth_by_image(targets, class_id)
    total_gt = int(sum(len(v) for v in gt_by_img.values()))
    if total_gt == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    matched = {img_idx: np.zeros((len(gt_boxes),), dtype=np.bool_) for img_idx, gt_boxes in gt_by_img.items()}
    tp = np.zeros((len(detections),), dtype=np.float32)
    fp = np.zeros((len(detections),), dtype=np.float32)

    for det_idx, (image_idx, _score, det_box) in enumerate(detections):
        gt_boxes = gt_by_img.get(image_idx, np.zeros((0, 4), dtype=np.float32))
        if len(gt_boxes) == 0:
            fp[det_idx] = 1.0
            continue

        ious = np.array([compute_iou(det_box, gt_box) for gt_box in gt_boxes], dtype=np.float32)
        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])
        if best_iou >= iou_threshold and not matched[image_idx][best_idx]:
            tp[det_idx] = 1.0
            matched[image_idx][best_idx] = True
        else:
            fp[det_idx] = 1.0

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / max(total_gt, 1)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
    return recalls, precisions


def _compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    changes = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def compute_precision_recall(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Tuple[float, float]:
    """Compute dataset precision/recall at a fixed IoU threshold."""
    num_classes = 0
    for target in targets:
        if target["labels"].numel() > 0:
            num_classes = max(num_classes, int(target["labels"].max().item()) + 1)
    for pred in predictions:
        if pred["labels"].numel() > 0:
            num_classes = max(num_classes, int(pred["labels"].max().item()) + 1)

    if num_classes == 0:
        return 0.0, 0.0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    for class_id in range(num_classes):
        recalls, precisions = _compute_pr_curve(predictions, targets, class_id=class_id, iou_threshold=iou_threshold)
        if recalls.size == 0:
            gt_count = int(sum((target["labels"] == class_id).sum().item() for target in targets))
            total_fn += float(gt_count)
            continue

        # Recover confusion counts from last cumulative point.
        last_recall = float(recalls[-1])
        last_precision = float(precisions[-1]) if precisions.size > 0 else 0.0
        gt_count = int(sum((target["labels"] == class_id).sum().item() for target in targets))
        tp = last_recall * gt_count
        fp = (tp / max(last_precision, 1e-12)) - tp
        fn = gt_count - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = float(total_tp / max(total_tp + total_fp, 1e-12))
    recall = float(total_tp / max(total_tp + total_fn, 1e-12))
    return precision, recall


def compute_ap(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    class_id: int,
    iou_threshold: float = 0.5
) -> float:
    """Compute AP for one class at a fixed IoU threshold."""
    recalls, precisions = _compute_pr_curve(predictions, targets, class_id=class_id, iou_threshold=iou_threshold)
    return _compute_ap_from_pr(recalls, precisions)


def compute_map(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Compute mAP and per-class AP at a fixed IoU threshold."""
    aps = [compute_ap(predictions, targets, class_id, iou_threshold) for class_id in range(num_classes)]
    return {"mAP": float(np.mean(aps)) if aps else 0.0, "per_class_AP": aps}


def _compute_prf1_at_iou(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float,
    fb_beta: float,
) -> Tuple[float, float, float]:
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    for class_id in range(num_classes):
        recalls, precisions = _compute_pr_curve(predictions, targets, class_id=class_id, iou_threshold=iou_threshold)
        gt_count = float(sum((target["labels"] == class_id).sum().item() for target in targets))
        if recalls.size == 0:
            total_fn += gt_count
            continue

        tp = float(recalls[-1]) * gt_count
        precision_last = float(precisions[-1]) if precisions.size > 0 else 0.0
        fp = (tp / max(precision_last, 1e-12)) - tp
        fn = gt_count - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / max(total_tp + total_fp, 1e-12)
    recall = total_tp / max(total_tp + total_fn, 1e-12)
    fb = (1.0+fb_beta**2) * precision * recall / max(fb_beta**2*precision + recall, 1e-12)
    return float(precision), float(recall), float(fb)


def compute_detection_metrics(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_thresholds: Iterable[float] | None = None,
) -> Dict[str, float]:
    """Compute common OD metrics from decoded predictions and targets.

    Args:
        predictions: List per image with keys ``boxes`` (xyxy), ``labels``, ``scores``.
        targets: List per image with keys ``boxes`` (xyxy), ``labels``.
        num_classes: Number of foreground classes.
        iou_thresholds: IoU thresholds for mAP50-95. Defaults to ``0.50:0.05:0.95``.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    iou_thresholds = [float(t) for t in iou_thresholds]
    if not iou_thresholds:
        iou_thresholds = [0.5]

    ap50_per_class = [compute_ap(predictions, targets, class_id=c, iou_threshold=0.5) for c in range(num_classes)]
    map50 = float(np.mean(ap50_per_class)) if ap50_per_class else 0.0

    mean_aps = []
    for threshold in iou_thresholds:
        aps = [compute_ap(predictions, targets, class_id=c, iou_threshold=threshold) for c in range(num_classes)]
        mean_aps.append(float(np.mean(aps)) if aps else 0.0)
    map50_95 = float(np.mean(mean_aps)) if mean_aps else 0.0

    precision50, recall50, f1_50 = _compute_prf1_at_iou(predictions, targets, num_classes=num_classes, iou_threshold=0.5, fb_beta=0.25)

    metrics: Dict[str, float] = {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision50": precision50,
        "recall50": recall50,
        "f1_50": f1_50,
        "num_gt_boxes": float(sum(target["boxes"].shape[0] for target in targets)),
        "num_pred_boxes": float(sum(pred["boxes"].shape[0] for pred in predictions)),
    }
    for class_id, ap in enumerate(ap50_per_class):
        metrics[f"AP50_class_{class_id}"] = float(ap)
    return metrics

