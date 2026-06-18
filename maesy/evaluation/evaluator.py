"""Evaluator for model evaluation."""

import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from .metrics import prepare_targets_for_detection_metrics, compute_iou

from ..dataset import MaesyDataset
from .metrics import compute_detection_metrics
from maesy.evaluation.inferer import Inferer
from ..model_tools.model_factory import create_model_from_checkpoint
from torchvision.transforms import v2 as transforms

from ..training.utils import collate_detection_fn


class Evaluator:
    """Evaluator for Vision Transformer object detection model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        dataset_path: str,
        device: str = "",
        split: str = "test",
        output_name: str = "",
    ):
        """
        Initialize evaluator.
        
        Args:
            :param checkpoint_path: Path to the checkpoint to evaluate
            :param dataset_path: Path to the MaeSyDataset with a test split to evaluate on
            :param device: Device to run evaluation on
            :param split: The dataset's split to evaluate on. Default: 'test'
            :param output_name: The output folder name for the results. Created as a subfolder in the checkpoint folder. Default: 'test_results/[dataset]/[checkpoint]'
        """
        if device == "":
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device=torch.device(device)

        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        self.dataset_name = os.path.basename(os.path.abspath(dataset_path))
        self.checkpoint_name = os.path.basename(os.path.abspath(checkpoint_path)).removesuffix('.pth')
        self.output_dir = os.path.join(checkpoint_dir, f"test_results/{self.dataset_name}/{self.checkpoint_name}" if output_name == "" else output_name)

        # Load model
        self.model = create_model_from_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()

        test_transforms = transforms.Compose( # TODO: make use of OD-Transforms?
            [  # TODO: make blank image folder possible again, "auto-infer" split? Maybe through 'None' -> All splits
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Resize((224, 224)),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        dataset = MaesyDataset(dataset_path, split, "detection", transforms=test_transforms, enable_lines=True, enable_ellipses=True)
        self.special_classes = dataset.get_special_classes()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_detection_fn)
        self.inferer = Inferer(model=self.model, data_loader=dataloader, device=device)
        self.class_labels = [
            dataset.id_to_name.get(i, str(i))
            for i in range(self.model.config.num_classes)
        ]
        self.class_name_slugs = [self._slugify_label(label) for label in self.class_labels]

    
    @torch.no_grad()
    def evaluate(
        self,
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.

        Returns:
            Dictionary with evaluation metrics
        """
        all_predictions, all_targets = self.inferer.infer(score_threshold=0.0)


        line_class_id = self.special_classes.get("line_class_id")
        ellipse_class_id = self.special_classes.get("ellipse_class_id")
        if line_class_id is not None and line_class_id < 0:
            line_class_id = None
        if ellipse_class_id is not None and ellipse_class_id < 0:
            ellipse_class_id = None

        prepared_targets = prepare_targets_for_detection_metrics(
            all_targets,
            line_class_id=line_class_id,
            ellipse_class_id=ellipse_class_id,
        )
        metrics = compute_detection_metrics(
            predictions=all_predictions,
            targets=prepared_targets,
            num_classes=self.model.config.num_classes,
            line_class_id=line_class_id,
            ellipse_class_id=ellipse_class_id,
            ellipse_shape_coef=getattr(self.model.config, "ellipse_shape_coef", 1.0),
        )

        os.makedirs(self.output_dir, exist_ok=True)
        curves = metrics.get("curves", {})
        if curves:
            plot_paths = self._save_curve_plots(curves)
            plot_paths.update(self._save_metric_grouped_plots(curves))
            plot_paths.update(self._save_line_ellipse_metric_grouped_plots(curves))
            metrics["plot_paths"] = plot_paths

        class_counts = self._count_dataset_classes(all_targets)
        counts_plot = self._plot_class_counts(class_counts)

        confusion_matrix = self._build_confusion_matrix(
            predictions=all_predictions,
            targets=prepared_targets,
            line_class_id=line_class_id,
            ellipse_class_id=ellipse_class_id,
            ellipse_shape_coef=getattr(self.model.config, "ellipse_shape_coef", 1.0),
        )
        confusion_plot = self._plot_confusion_matrix(confusion_matrix)
        confusion_csv = self._save_confusion_matrix_csv(confusion_matrix)

        metrics_to_print = self._filter_metrics_for_output(metrics)
        self._print_and_save_metrics(self.dataset_name, self.checkpoint_name, metrics_to_print, class_counts)

        return metrics

    def _save_curve_plots(self, curves: Dict[str, Any]) -> Dict[str, str]:
        plot_paths: Dict[str, str] = {}

        if "bbox" in curves:
            combined = curves["bbox"]["combined"]
            plot_paths.update(self._plot_pr_and_confidence(
                combined,
                prefix="bbox_combined",
                title_prefix="BBox Combined",
            ))
            for class_id, data in curves["bbox"]["per_class"].items():
                plot_paths.update(self._plot_pr_and_confidence(
                    data,
                    prefix=f"bbox_class_{self.class_name_slugs[class_id]}",
                    title_prefix=f"BBox {self.class_labels[class_id]}",
                ))

        if "line" in curves:
            for threshold, data in curves["line"].items():
                plot_paths.update(self._plot_pr_and_confidence(
                    data,
                    prefix=f"line_{threshold}",
                    title_prefix=f"Line @ {threshold}",
                ))

        if "ellipse" in curves:
            for threshold, data in curves["ellipse"].items():
                plot_paths.update(self._plot_pr_and_confidence(
                    data,
                    prefix=f"ellipse_{threshold}",
                    title_prefix=f"Ellipse @ {threshold}",
                ))

        return plot_paths

    def _save_metric_grouped_plots(self, curves: Dict[str, Any]) -> Dict[str, str]:
        plot_paths: Dict[str, str] = {}
        bbox_curves = curves.get("bbox")
        if not bbox_curves:
            return plot_paths

        per_class = bbox_curves.get("per_class", {})
        combined = bbox_curves.get("combined")
        if not per_class or combined is None:
            return plot_paths

        sorted_class_ids = sorted(per_class.keys())
        plot_paths["bbox_metric_pr"] = self._plot_pr_by_class(
            per_class=per_class,
            combined=combined,
            class_ids=sorted_class_ids,
            title="BBox Precision-Recall (All Classes)",
            filename="bbox_metric_pr.svg",
        )

        combined_conf = combined.get("confidence", {})
        thresholds = combined_conf.get("thresholds")
        if thresholds is None or len(thresholds) == 0:
            return plot_paths

        metric_keys = [
            key for key in combined_conf.keys()
            if key != "thresholds"
        ]
        for metric_key in metric_keys:
            plot_paths[f"bbox_metric_{metric_key}"] = self._plot_confidence_by_class(
                per_class=per_class,
                combined=combined,
                class_ids=sorted_class_ids,
                metric_key=metric_key,
                title=f"BBox {metric_key} vs Confidence (All Classes)",
                filename=f"bbox_metric_{metric_key}.svg",
            )

        return plot_paths

    def _save_line_ellipse_metric_grouped_plots(self, curves: Dict[str, Any]) -> Dict[str, str]:
        plot_paths: Dict[str, str] = {}

        line_curves = curves.get("line", {})
        if line_curves:
            plot_paths["line_metric_pr"] = self._plot_pr_by_threshold(
                curves_by_threshold=line_curves,
                title="Line Precision-Recall (All Thresholds)",
                filename="line_metric_pr.svg",
            )
            plot_paths.update(self._plot_confidence_metrics_by_threshold(
                curves_by_threshold=line_curves,
                prefix="line_metric",
            ))

        ellipse_curves = curves.get("ellipse", {})
        if ellipse_curves:
            plot_paths["ellipse_metric_pr"] = self._plot_pr_by_threshold(
                curves_by_threshold=ellipse_curves,
                title="Ellipse Precision-Recall (All Thresholds)",
                filename="ellipse_metric_pr.svg",
            )
            plot_paths.update(self._plot_confidence_metrics_by_threshold(
                curves_by_threshold=ellipse_curves,
                prefix="ellipse_metric",
            ))

        return plot_paths

    def _plot_pr_and_confidence(self, curve_data: Dict[str, Any], prefix: str, title_prefix: str) -> Dict[str, str]:
        plot_paths: Dict[str, str] = {}
        pr = curve_data.get("pr", {})
        confidence = curve_data.get("confidence", {})

        recall = pr.get("recall")
        precision = pr.get("precision")
        if recall is not None and precision is not None and len(recall) > 0:
            plot_paths[f"{prefix}_pr"] = self._plot_pr_curve(
                recall,
                precision,
                title=f"{title_prefix} Precision-Recall",
                filename=f"{prefix}_pr.svg",
            )

        thresholds = confidence.get("thresholds")
        if thresholds is not None and len(thresholds) > 0:
            fb_key = next((k for k in confidence.keys() if k.startswith("f") and k != "f1"), None)
            plot_paths[f"{prefix}_confidence"] = self._plot_confidence_curves(
                thresholds=thresholds,
                precision=confidence.get("precision"),
                recall=confidence.get("recall"),
                f1=confidence.get("f1"),
                fb=confidence.get(fb_key) if fb_key else None,
                fb_label=fb_key or "fb",
                title=f"{title_prefix} Confidence Curves",
                filename=f"{prefix}_confidence.svg",
            )
        return plot_paths

    def _plot_pr_curve(self, recall, precision, title: str, filename: str) -> str:
        plt.figure(figsize=(5, 4))
        plt.plot(recall, precision, color="tab:blue", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, format="svg")
        plt.close()
        return path

    def _plot_confidence_curves(
        self,
        thresholds,
        precision,
        recall,
        f1,
        fb,
        fb_label: str,
        title: str,
        filename: str,
    ) -> str:
        plt.figure(figsize=(6, 4))
        if precision is not None:
            plt.plot(thresholds, precision, label="Precision")
        if recall is not None:
            plt.plot(thresholds, recall, label="Recall")
        if f1 is not None:
            plt.plot(thresholds, f1, label="F1")
        if fb is not None:
            plt.plot(thresholds, fb, label=fb_label)
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Score")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, format="svg")
        plt.close()
        return path

    def _plot_pr_by_class(
        self,
        per_class: Dict[int, Dict[str, Any]],
        combined: Dict[str, Any],
        class_ids: list[int],
        title: str,
        filename: str,
    ) -> str:
        plt.figure(figsize=(6, 5))
        for class_id in class_ids:
            pr = per_class[class_id].get("pr", {})
            recall = pr.get("recall")
            precision = pr.get("precision")
            if recall is None or precision is None or len(recall) == 0:
                continue
            plt.plot(recall, precision, linewidth=1.2, label=self.class_labels[class_id])

        combined_pr = combined.get("pr", {})
        combined_recall = combined_pr.get("recall")
        combined_precision = combined_pr.get("precision")
        if combined_recall is not None and combined_precision is not None and len(combined_recall) > 0:
            plt.plot(combined_recall, combined_precision, color="black", linewidth=2.0, label="Combined")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, format="svg")
        plt.close()
        return path

    def _plot_confidence_by_class(
        self,
        per_class: Dict[int, Dict[str, Any]],
        combined: Dict[str, Any],
        class_ids: list[int],
        metric_key: str,
        title: str,
        filename: str,
    ) -> str:
        plt.figure(figsize=(6, 4))

        combined_conf = combined.get("confidence", {})
        thresholds = combined_conf.get("thresholds")
        combined_metric = combined_conf.get(metric_key)
        if thresholds is not None and combined_metric is not None:
            plt.plot(thresholds, combined_metric, color="black", linewidth=2.0, label="Combined")

        for class_id in class_ids:
            conf = per_class[class_id].get("confidence", {})
            class_thresholds = conf.get("thresholds")
            class_metric = conf.get(metric_key)
            if class_thresholds is None or class_metric is None:
                continue
            plt.plot(class_thresholds, class_metric, linewidth=1.2, label=self.class_labels[class_id])

        plt.xlabel("Confidence Threshold")
        plt.ylabel(metric_key)
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, format="svg")
        plt.close()
        return path

    def _plot_pr_by_threshold(
        self,
        curves_by_threshold: Dict[str, Dict[str, Any]],
        title: str,
        filename: str,
    ) -> str:
        plt.figure(figsize=(6, 5))
        for threshold, data in curves_by_threshold.items():
            pr = data.get("pr", {})
            recall = pr.get("recall")
            precision = pr.get("precision")
            if recall is None or precision is None or len(recall) == 0:
                continue
            plt.plot(recall, precision, linewidth=1.2, label=f"thr={threshold}")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, format="svg")
        plt.close()
        return path

    def _plot_confidence_metrics_by_threshold(
        self,
        curves_by_threshold: Dict[str, Dict[str, Any]],
        prefix: str,
    ) -> Dict[str, str]:
        plot_paths: Dict[str, str] = {}
        metric_keys = set()
        for data in curves_by_threshold.values():
            confidence = data.get("confidence", {})
            metric_keys.update(key for key in confidence.keys() if key != "thresholds")

        for metric_key in sorted(metric_keys):
            plt.figure(figsize=(6, 4))
            for threshold, data in curves_by_threshold.items():
                confidence = data.get("confidence", {})
                thresholds = confidence.get("thresholds")
                metric = confidence.get(metric_key)
                if thresholds is None or metric is None:
                    continue
                plt.plot(thresholds, metric, linewidth=1.2, label=f"thr={threshold}")
            plt.xlabel("Confidence Threshold")
            plt.ylabel(metric_key)
            plt.title(f"{prefix.replace('_', ' ').title()} {metric_key}")
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            filename = f"{prefix}_{metric_key}.svg"
            path = os.path.join(self.output_dir, filename)
            plt.tight_layout()
            plt.savefig(path, format="svg")
            plt.close()
            plot_paths[f"{prefix}_{metric_key}"] = path

        return plot_paths

    def _filter_metrics_for_output(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value for key, value in metrics.items()
            if key not in {"curves", "plot_paths"}
        }

    def _print_and_save_metrics(self, dataset_name: str, checkpoint_name: str, metrics: Dict[str, Any], class_counts: Dict[str, int]) -> None:
        lines: list[str] = []
        lines.append("Evaluation Metrics")
        lines.append("=" * 72)
        lines.append(f"{'Dataset:':32s} {dataset_name}")
        lines.append(f"{'Checkpoint:':32s} {checkpoint_name}")
        lines.append("=" * 72)
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if isinstance(value, float):
                lines.append(f"{key:32s}: {value:.4f}")
            else:
                lines.append(f"{key:32s}: {value}")
        lines.append("")
        lines.append("Class Counts")
        lines.append("-" * 72)
        for name, count in class_counts.items():
            lines.append(f"{name:32s}: {count}")

        report = "\n".join(lines)
        print(report)

        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, "evaluation_metrics.txt")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(report + "\n")

    def _count_dataset_classes(self, targets: list[Dict[str, torch.Tensor]]) -> Dict[str, int]:
        counts = {label: 0 for label in self.class_labels}
        for target in targets:
            labels = target.get("labels", torch.empty((0,), dtype=torch.long))
            for label in labels.tolist():
                if 0 <= label < len(self.class_labels):
                    counts[self.class_labels[label]] += 1
        return counts

    def _plot_class_counts(self, class_counts: Dict[str, int]) -> str:
        labels = list(class_counts.keys())
        values = [class_counts[label] for label in labels]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values, color="tab:blue")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Test Dataset Class Counts")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "class_counts.svg")
        plt.savefig(path, format="svg")
        plt.close()
        return path

    def _build_confusion_matrix(
        self,
        predictions: list[Dict[str, torch.Tensor]],
        targets: list[Dict[str, torch.Tensor]],
        line_class_id: int | None,
        ellipse_class_id: int | None,
        ellipse_shape_coef: float,
        iou_threshold: float = 0.5,
        line_distance_threshold: float = 0.05,
        ellipse_distance_threshold: float = 0.1,
    ) -> np.ndarray:
        matrix = np.zeros((len(self.class_labels), len(self.class_labels)), dtype=np.float32)
        for image_idx, (pred, target) in enumerate(zip(predictions, targets)):
            pred_boxes = pred.get("boxes", torch.empty((0, 4)))
            pred_labels = pred.get("labels", torch.empty((0,), dtype=torch.long))
            pred_scores = pred.get("scores", torch.empty((0,)))
            gt_boxes = target.get("boxes", torch.empty((0, 4)))
            gt_labels = target.get("labels", torch.empty((0,), dtype=torch.long))

            if pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
                scores_sorted = torch.argsort(pred_scores, descending=True)
                matched_gt = np.zeros((gt_boxes.shape[0],), dtype=np.bool_)
                for pred_idx in scores_sorted.tolist():
                    pred_box = pred_boxes[pred_idx].numpy()
                    ious = np.array([compute_iou(pred_box, gt_box.numpy()) for gt_box in gt_boxes], dtype=np.float32)
                    best_idx = int(np.argmax(ious))
                    if ious[best_idx] >= iou_threshold and not matched_gt[best_idx]:
                        gt_label = int(gt_labels[best_idx].item())
                        pred_label = int(pred_labels[pred_idx].item())
                        if 0 <= gt_label < matrix.shape[0] and 0 <= pred_label < matrix.shape[1]:
                            matrix[gt_label, pred_label] += 1
                        matched_gt[best_idx] = True
            if line_class_id is not None:
                pred_lines = pred.get("line_points", torch.empty((0, 4)))
                pred_line_labels = pred.get("line_labels", torch.empty((0,), dtype=torch.long))
                pred_line_scores = pred.get("line_scores", torch.empty((0,)))
                gt_lines = target.get("line_points", torch.empty((0, 4)))
                gt_line_labels = target.get("line_labels", torch.empty((0,), dtype=torch.long))
                if pred_lines.numel() > 0 and gt_lines.numel() > 0:
                    scores_sorted = torch.argsort(pred_line_scores, descending=True)
                    matched_gt = np.zeros((gt_lines.shape[0],), dtype=np.bool_)
                    for pred_idx in scores_sorted.tolist():
                        pred_line = pred_lines[pred_idx].numpy()
                        distances = np.array([
                            self._line_endpoint_distance(pred_line, gt_line.numpy()) for gt_line in gt_lines
                        ], dtype=np.float32)
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= line_distance_threshold and not matched_gt[best_idx]:
                            gt_label = int(gt_line_labels[best_idx].item())
                            pred_label = int(pred_line_labels[pred_idx].item())
                            if 0 <= gt_label < matrix.shape[0] and 0 <= pred_label < matrix.shape[1]:
                                matrix[gt_label, pred_label] += 1
                            matched_gt[best_idx] = True
            if ellipse_class_id is not None:
                pred_ellipses = pred.get("ellipses", torch.empty((0, 6)))
                pred_ellipse_labels = pred.get("ellipse_labels", torch.empty((0,), dtype=torch.long))
                pred_ellipse_scores = pred.get("ellipse_scores", torch.empty((0,)))
                gt_ellipses = target.get("ellipses", torch.empty((0, 6)))
                gt_ellipse_labels = target.get("ellipse_labels", torch.empty((0,), dtype=torch.long))
                if pred_ellipses.numel() > 0 and gt_ellipses.numel() > 0:
                    scores_sorted = torch.argsort(pred_ellipse_scores, descending=True)
                    matched_gt = np.zeros((gt_ellipses.shape[0],), dtype=np.bool_)
                    for pred_idx in scores_sorted.tolist():
                        pred_ellipse = pred_ellipses[pred_idx].numpy()
                        distances = np.array([
                            self._ellipse_distance(pred_ellipse, gt_ellipse.numpy(), ellipse_shape_coef)
                            for gt_ellipse in gt_ellipses
                        ], dtype=np.float32)
                        best_idx = int(np.argmin(distances))
                        if distances[best_idx] <= ellipse_distance_threshold and not matched_gt[best_idx]:
                            gt_label = int(gt_ellipse_labels[best_idx].item())
                            pred_label = int(pred_ellipse_labels[pred_idx].item())
                            if 0 <= gt_label < matrix.shape[0] and 0 <= pred_label < matrix.shape[1]:
                                matrix[gt_label, pred_label] += 1
                            matched_gt[best_idx] = True

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return matrix / row_sums

    @staticmethod
    def _line_endpoint_distance(line_a: np.ndarray, line_b: np.ndarray) -> float:
        a0 = line_a[:2]
        a1 = line_a[2:]
        b0 = line_b[:2]
        b1 = line_b[2:]
        direct = (np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1)) * 0.5
        swapped = (np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0)) * 0.5
        return float(min(direct, swapped))

    @staticmethod
    def _ellipse_distance(ellipse_a: np.ndarray, ellipse_b: np.ndarray, shape_coef: float) -> float:
        center = np.abs(ellipse_a[:2] - ellipse_b[:2]).sum()
        shape = np.abs(ellipse_a[2:4] - ellipse_b[2:4]).sum()
        rotation = float((ellipse_a[4] - ellipse_b[4]) ** 2 + (ellipse_a[5] - ellipse_b[5]) ** 2)
        return float(center + shape_coef * (shape + rotation))

    def _plot_confusion_matrix(self, matrix: np.ndarray) -> str:
        plt.figure(figsize=(7, 6))
        plt.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        plt.title("Confusion Matrix (Normalized)")
        plt.colorbar()
        tick_marks = np.arange(len(self.class_labels))
        plt.xticks(tick_marks, self.class_labels, rotation=45, ha="right")
        plt.yticks(tick_marks, self.class_labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                color = "white" if value >= 0.5 else "black"
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "confusion_matrix_normalized.svg")
        plt.savefig(path, format="svg")
        plt.close()
        return path

    def _save_confusion_matrix_csv(self, matrix: np.ndarray) -> str:
        path = os.path.join(self.output_dir, "confusion_matrix_normalized.csv")
        header = ",".join(["gt\\pred"] + self.class_labels)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(header + "\n")
            for label, row in zip(self.class_labels, matrix):
                row_str = ",".join(f"{value:.6f}" for value in row)
                handle.write(f"{label},{row_str}\n")
        return path

    @staticmethod
    def _slugify_label(label: str) -> str:
        slug = "".join(ch if ch.isalnum() else "_" for ch in label.strip())
        slug = slug.strip("_")
        return slug if slug else "label"
