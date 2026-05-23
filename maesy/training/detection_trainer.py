"""Trainer for Object Detection models."""

from typing import Optional, List, Dict

import cv2
import torch
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

from maesy.evaluation.metrics import (
    compute_detection_metrics,
    decode_detr_predictions,
    prepare_targets_for_detection_metrics,
)
from maesy.evaluation.visualizer import draw_objects_in_tensor
from maesy.training import BaseTrainer


class DetectionTrainer(BaseTrainer):
    """Trainer for Object Detection Vision Transformer models.

    This trainer handles the training loop, loss computation, and optimization
    specific to object detection models using the DetectionLoss.
    """

    def _validation_start(self):
        self._val_predictions: List[Dict[str, torch.Tensor]] = []
        self._val_targets: List[Dict[str, torch.Tensor]] = []

    def _validation_step(self, images: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]], losses: Dict[str, torch.Tensor]):
        if targets is None:
            return
        decoded = losses.get("__decoded_predictions")
        prepared_targets = losses.get("__prepared_targets")
        if decoded is not None:
            self._val_predictions.extend(decoded)
        if prepared_targets is not None:
            self._val_targets.extend(prepared_targets)

    def _validation_finalize(self) -> Dict[str, float]:
        if len(self._val_predictions) == 0 or len(self._val_targets) == 0:
            return {}
        line_class_id = getattr(self.model.config, "line_class_id", None)
        if line_class_id is not None and line_class_id < 0:
            line_class_id = None
        return compute_detection_metrics(
            predictions=self._val_predictions,
            targets=self._val_targets,
            num_classes=self.model.config.num_classes,
            line_class_id=line_class_id,
        )

    # def _render_all_query_predictions(
    #     self,
    #     image: torch.Tensor,
    #     pred_boxes: torch.Tensor,
    #     pred_lines: torch.Tensor,
    #     pred_logits: torch.Tensor,
    #     special_classes: Dict[str, int],
    #     draw_no_obj: bool = False
    # ) -> torch.Tensor:
    #     # Unnormalize to RGB uint8 so draw_bounding_boxes always renders visible overlays.
    #     img = image.detach().cpu()
    #     img = (img * self._IMAGENET_STD) + self._IMAGENET_MEAN
    #     img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    #
    #     probs = pred_logits.detach().cpu().softmax(-1)
    #     scores, labels = probs.max(dim=-1)
    #     no_obj = pred_logits.shape[-1] - 1
    #
    #     confidence_threshold = 0.3
    #     # filter predictions below the confidence_threshold
    #     keep = scores >= confidence_threshold
    #     pred_boxes = pred_boxes[keep]
    #     pred_lines = pred_lines[keep] # TODO: Maybe this fails if lines are deactivated??
    #     labels = labels[keep]
    #     scores = scores[keep]
    #
    #     # Separate into bboxes and other types
    #     # build boolean mask: keep boxes whose label is not in special_classes values
    #     special_vals = list(special_classes.values())
    #     if not draw_no_obj:
    #         special_vals.append(no_obj)
    #
    #     if len(special_vals) == 0:
    #         bbox_idx = torch.ones(labels.shape, dtype=torch.bool)
    #     else:
    #         special_tensor = torch.tensor(special_vals, dtype=labels.dtype)
    #         bbox_idx = ~torch.isin(labels, special_tensor)
    #     boxes = pred_boxes[bbox_idx]
    #     labels_bbox = labels[bbox_idx]
    #
    #     h, w = img.shape[-2:]
    #
    #     if len(boxes) > 0:
    #         boxes_xyxy = box_convert(boxes.detach().cpu(), "cxcywh", "xyxy")
    #         boxes_xyxy[:, (0, 2)] *= w
    #         boxes_xyxy[:, (1, 3)] *= h
    #         boxes_xyxy[:, (0, 2)] = boxes_xyxy[:, (0, 2)].clamp(0, w - 1)
    #         boxes_xyxy[:, (1, 3)] = boxes_xyxy[:, (1, 3)].clamp(0, h - 1)
    #
    #         rendered_labels: List[str] = []
    #         colors: List[str] = []
    #         for query_idx, (cls_id, score) in enumerate(zip(labels_bbox.tolist(), scores.tolist())):
    #             cls_name = "NoObj" if cls_id == no_obj else f"C{cls_id}"
    #             rendered_labels.append(f"{cls_name} {score:.2f}")
    #             colors.append("gray" if cls_id == no_obj else "blue")
    #
    #         drawn = draw_bounding_boxes(img, boxes_xyxy, labels=rendered_labels, colors=colors, width=1)
    #     else:
    #         print("No Boxes in Image, directly using drawn=img and converting to uint8")
    #         drawn = img.detach().cpu().clamp(0, 255).to(torch.uint8)
    #         # print("Drawn: ", drawn)
    #         # print("Drawn.shape: ", drawn.shape)
    #
    #     drawn = drawn.permute(1, 2, 0).contiguous().numpy()
    #     for k, v in special_classes.items():
    #         special_ids = labels == v
    #         objs = pred_lines[special_ids]
    #
    #         if k == "line_class_id":
    #             for line in objs:
    #                 x1, y1, x2, y2 = line.detach().cpu()
    #                 x1 = int(x1.item() * w)
    #                 y1 = int(y1.item() * h)
    #                 x2 = int(x2.item() * w)
    #                 y2 = int(y2.item() * h)
    #                 cv2.line(drawn, (x1, y1), (x2, y2), color=(255, 0, 255), thickness=2) # cv2.line draws in-place !!!
    #     drawn = torch.from_numpy(drawn).permute(2, 0, 1).contiguous()
    #
    #     return drawn.float() / 255.0

    def _render_target_boxes(self, image: torch.Tensor, target: Dict[str, torch.Tensor]) -> torch.Tensor:
        img = image.detach().cpu()
        img = (img * self._IMAGENET_STD) + self._IMAGENET_MEAN
        img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

        boxes = target["boxes"].detach().cpu()
        labels = target["labels"].detach().cpu()
        if boxes.numel() == 0:
            return img.float() / 255.0

        h, w = img.shape[-2:]
        boxes_xyxy = box_convert(boxes, "cxcywh", "xyxy")
        boxes_xyxy[:, (0, 2)] *= w
        boxes_xyxy[:, (1, 3)] *= h
        boxes_xyxy[:, (0, 2)] = boxes_xyxy[:, (0, 2)].clamp(0, w - 1)
        boxes_xyxy[:, (1, 3)] = boxes_xyxy[:, (1, 3)].clamp(0, h - 1)

        rendered_labels = [f"GT:C{label.item()}" for label in labels]
        drawn = draw_bounding_boxes(img, boxes_xyxy, labels=rendered_labels, colors="green", width=2)
        return drawn.float() / 255.0

    def _forward_model(self, images: torch.Tensor, targets: Optional[List[Dict[str, torch.Tensor]]], val: bool) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            images: Input images [B, C, H, W]
            targets: List of target dictionaries containing 'boxes' and 'labels'
            val: Whether this is a validation pass

        Returns:
            Dictionary of losses from the loss function
        """
        # Get model predictions
        predictions = self.model.forward(images, targets=targets)

        # Compute loss
        losses = self.loss(predictions, targets)

        if val and targets is not None:
            line_class_id: int | None = getattr(self.model.config, "line_class_id", None)
            if line_class_id is not None and line_class_id < 0:
                line_class_id = None
            ellipse_class_id: int | None = getattr(self.model.config, "ellipse_class_id", None)
            if ellipse_class_id is not None and ellipse_class_id < 0:
                ellipse_class_id = None

            predictions = decode_detr_predictions( # TODO: Move this to postprocessing of rt-detr
                pred_logits=predictions["pred_logits"],
                pred_boxes=predictions["pred_boxes"],
                pred_lines=predictions.get("pred_lines"),
                pred_ellipses=predictions.get("pred_ellipses"),
                line_class_id=line_class_id,
                ellipse_class_id=ellipse_class_id,
                no_object_class=predictions["pred_logits"].shape[-1] - 1,
                score_threshold=0.5,
            )
            losses["__decoded_predictions"] = predictions

            losses["__prepared_targets"] = prepare_targets_for_detection_metrics(
                targets,
                line_class_id=line_class_id,
            )
            # losses["img_queries_all"] = self._render_all_query_predictions(
            #     image=images[0],
            #     pred_boxes=predictions["pred_boxes"][0],
            #     pred_lines=predictions["pred_lines"][0],
            #     pred_logits=predictions["pred_logits"][0],
            #     special_classes ={"line_class_id": self.model.config.line_class_id}
            # )
            # losses["img_targets"] = self._render_target_boxes(images[0], targets[0])

            # Unnormalize to RGB uint8 so draw_bounding_boxes always renders visible overlays.
            img = images[0].detach()
            img = (img * self._IMAGENET_STD) + self._IMAGENET_MEAN
            img = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu()
            losses["img_predictions"] = draw_objects_in_tensor(img, predictions[0]["boxes"], predictions[0]["labels"], predictions[0]["line_points"], predictions[0]["ellipses"], xyxy=True)/255

        return losses