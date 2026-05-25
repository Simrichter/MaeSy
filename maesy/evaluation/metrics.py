"""Evaluation metrics for object detection."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torchvision.ops import box_convert

from maesy.dataset import sanitize_cxcywh


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

def prepare_targets_for_detection_metrics(
    targets: List[Dict[str, torch.Tensor]],
    line_class_id: int | None = None,
) -> List[Dict[str, torch.Tensor]]:
    """
        Convert training targets from normalized ``cxcywh`` to ``xyxy``.
        Also sanitizes boxes and lines
    """
    prepared: List[Dict[str, torch.Tensor]] = []
    for target in targets:
        boxes = target.get("boxes", torch.empty((0, 4))).detach().cpu().float()
        labels = target.get("labels", torch.empty((0,), dtype=torch.long)).detach().cpu().long()
        line_points = target.get("line_points", torch.empty((0, 4))).detach().cpu().float()

        num_boxes = int(boxes.shape[0])
        num_lines = int(line_points.shape[0])

        if labels.shape[0] >= num_boxes:
            box_labels = labels[:num_boxes]
        else:
            box_labels = labels

        if line_class_id is not None and labels.shape[0] >= (num_boxes + num_lines):
            line_labels = labels[num_boxes:num_boxes + num_lines]
        elif line_class_id is not None and num_lines > 0:
            line_labels = torch.full((num_lines,), line_class_id, dtype=torch.long)
        else:
            line_labels = torch.empty((0,), dtype=torch.long)

        boxes_xyxy, valid = sanitize_cxcywh(boxes) # ! returns xyxy format
        if boxes.numel() == 0:
            box_labels = torch.empty((0,), dtype=torch.long)
        else:
            box_labels = box_labels[valid]

        line_points = line_points.clamp(0.0, 1.0)
        line_valid = (line_points[:, :2] - line_points[:, 2:]).abs().sum(dim=1) > 0
        line_points = line_points[line_valid]
        if line_labels.numel() > 0:
            line_labels = line_labels[line_valid]

        prepared.append({
            "boxes": boxes_xyxy,
            "labels": box_labels,
            "line_points": line_points,
            "line_labels": line_labels,
        })
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
        ground_truth[image_idx] = target["boxes"][class_mask].cpu().numpy().astype(np.float32)
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


def compute_ap(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    class_id: int,
    iou_threshold: float = 0.5
) -> float:
    """Compute AP for one class at a fixed IoU threshold."""
    recalls, precisions = _compute_pr_curve(predictions, targets, class_id=class_id, iou_threshold=iou_threshold)
    return _compute_ap_from_pr(recalls, precisions)


def _compute_prf1_at_iou(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    class_ids: Sequence[int],
    iou_threshold: float,
    fb_beta: float,
) -> Tuple[float, float, float]:
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    for class_id in class_ids:
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


def _line_endpoint_distance(line_a: np.ndarray, line_b: np.ndarray) -> float:
    a0 = line_a[:2]
    a1 = line_a[2:]
    b0 = line_b[:2]
    b1 = line_b[2:]
    direct = (np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1)) * 0.5
    swapped = (np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0)) * 0.5
    return float(min(direct, swapped))


def _compute_line_pr_curve(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    distance_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    detections: List[Tuple[int, float, np.ndarray]] = []
    gt_by_img: Dict[int, np.ndarray] = {}

    for image_idx, pred in enumerate(predictions):
        line_points = pred.get("line_points", torch.empty((0, 4)))
        line_scores = pred.get("line_scores", torch.empty((0,)))
        if line_points.numel() > 0:
            for line, score in zip(line_points.numpy(), line_scores.numpy()):
                detections.append((image_idx, float(score), line.astype(np.float32)))

        gt_lines = targets[image_idx].get("line_points", torch.empty((0, 4)))
        gt_by_img[image_idx] = gt_lines.numpy().astype(np.float32) if gt_lines.numel() > 0 else np.zeros((0, 4), dtype=np.float32)

    detections.sort(key=lambda item: item[1], reverse=True)

    total_gt = int(sum(len(v) for v in gt_by_img.values()))
    if total_gt == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty

    matched = {img_idx: np.zeros((len(gt_lines),), dtype=np.bool_) for img_idx, gt_lines in gt_by_img.items()}
    tp = np.zeros((len(detections),), dtype=np.float32)
    fp = np.zeros((len(detections),), dtype=np.float32)
    matched_distances: List[float] = []

    for det_idx, (image_idx, _score, det_line) in enumerate(detections):
        gt_lines = gt_by_img.get(image_idx, np.zeros((0, 4), dtype=np.float32))
        if len(gt_lines) == 0:
            fp[det_idx] = 1.0
            continue

        distances = np.array([_line_endpoint_distance(det_line, gt_line) for gt_line in gt_lines], dtype=np.float32)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])

        if best_dist <= distance_threshold and not matched[image_idx][best_idx]:
            tp[det_idx] = 1.0
            matched[image_idx][best_idx] = True
            matched_distances.append(best_dist)
        else:
            fp[det_idx] = 1.0

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    recalls = tp_cumsum / max(total_gt, 1)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
    return recalls, precisions, np.array(matched_distances, dtype=np.float32)


def compute_detection_metrics(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_thresholds: Iterable[float] | None = None,
    line_class_id: int | None = None,
    line_distance_thresholds: Sequence[float] | None = None,
    fb_beta=0.25
) -> Dict[str, float]:
    """Compute common OD metrics from decoded predictions and targets.

        Args:
            :param predictions: List per image with keys ``boxes`` (xyxy), ``labels``, ``scores``.
            :param targets: List per image with keys ``boxes`` (xyxy), ``labels``.
            :param num_classes: Number of foreground classes.
            :param iou_thresholds: IoU thresholds for mAP50-95. Defaults to ``0.50:0.05:0.95``.
            :param line_class_id: ID of the line class. None for training without lines
            :param line_distance_thresholds: A sequence of thresholds for max line distances
            :param fb_beta: The weighting for the F-Score. Defaults to 0.25 (to weigh precision 4x)
        """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    iou_thresholds = [float(t) for t in iou_thresholds]
    if not iou_thresholds:
        iou_thresholds = [0.5]

    eval_class_ids = [c for c in range(num_classes) if line_class_id is None or c != line_class_id]

    ap50_per_class = [compute_ap(predictions, targets, class_id=c, iou_threshold=0.5) for c in eval_class_ids]
    map50 = float(np.mean(ap50_per_class)) if ap50_per_class else 0.0

    mean_aps = []
    for threshold in iou_thresholds:
        aps = [compute_ap(predictions, targets, class_id=c, iou_threshold=threshold) for c in eval_class_ids]
        mean_aps.append(float(np.mean(aps)) if aps else 0.0)
    map50_95 = float(np.mean(mean_aps)) if mean_aps else 0.0

    precision50, recall50, fb_50 = _compute_prf1_at_iou(
        predictions,
        targets,
        class_ids=eval_class_ids,
        iou_threshold=0.5,
        fb_beta=fb_beta,
    ) if eval_class_ids else (0.0, 0.0, 0.0)

    metrics: Dict[str, float] = {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision50": precision50,
        "recall50": recall50,
        f"f{fb_beta}_50": fb_50,
        "num_gt_boxes": float(sum(target["boxes"].shape[0] for target in targets)),
        "num_pred_boxes": float(sum(pred["boxes"].shape[0] for pred in predictions)),
    }
    for class_id, ap in zip(eval_class_ids, ap50_per_class):
        metrics[f"AP50_class_{class_id}"] = float(ap)

    metrics["total_mAP"] = map50_95 # Line mAP and Ellipse mAP are added later

    has_line_data = any(target.get("line_points", torch.empty((0, 4))).numel() > 0 for target in targets)
    if line_class_id is not None and has_line_data:
        if line_distance_thresholds is None:
            line_distance_thresholds = (0.02, 0.05, 0.1)
        thresholds = [float(t) for t in line_distance_thresholds]
        if not thresholds:
            thresholds = [0.05]

        line_ap_values: List[float] = []
        for threshold in thresholds:
            line_recalls, line_precisions, matched_distances = _compute_line_pr_curve(
                predictions,
                targets,
                distance_threshold=threshold,
            )
            line_ap = _compute_ap_from_pr(line_recalls, line_precisions)
            line_ap_values.append(line_ap)
            precision_thr = float(line_precisions[-1]) if line_precisions.size > 0 else 0.0
            recall_thr = float(line_recalls[-1]) if line_recalls.size > 0 else 0.0
            f1_thr = (2.0 * precision_thr * recall_thr) / max(precision_thr + recall_thr, 1e-12)
            metrics[f"line_precision@{threshold:.2f}"] = precision_thr
            metrics[f"line_recall@{threshold:.2f}"] = recall_thr
            metrics[f"line_f1@{threshold:.2f}"] = float(f1_thr)
            metrics[f"line_AP@{threshold:.2f}"] = float(line_ap)
            metrics[f"line_endpoint_error@{threshold:.2f}"] = float(matched_distances.mean()) if matched_distances.size > 0 else 0.0

        metrics["line_mAP"] = float(np.mean(line_ap_values)) if line_ap_values else 0.0
        metrics["num_gt_lines"] = float(sum(target.get("line_points", torch.empty((0, 4))).shape[0] for target in targets))
        metrics["num_pred_lines"] = float(sum(pred.get("line_points", torch.empty((0, 4))).shape[0] for pred in predictions))

        metrics["total_mAP"] += metrics["line_mAP"]

    return metrics

