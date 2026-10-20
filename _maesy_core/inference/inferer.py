from collections.abc import Mapping
import math
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from _maesy_core.dataset import sanitize_cxcywh
from _maesy_core.dataset.utils import handle_raw_batch
from _maesy_core.model import BaseModel
from _maesy_core.utils import to_device


_RAW_DETECTION_KEYS = {"pred_logits", "pred_boxes"}
_DECODED_DETECTION_KEYS = {"boxes", "labels", "scores"}


def _looks_like_decoded_prediction(prediction) -> bool:
    return isinstance(prediction, Mapping) and bool(
        _DECODED_DETECTION_KEYS.intersection(prediction)
    )


def _validate_decoded_prediction(prediction, image_index: int) -> None:
    """Check the per-image detection structure consumed by evaluation metrics."""
    missing = _DECODED_DETECTION_KEYS.difference(prediction)
    if missing:
        raise ValueError(
            f"Detection prediction for image {image_index} is missing "
            f"required fields: {', '.join(sorted(missing))}"
        )

    boxes = prediction["boxes"]
    labels = prediction["labels"]
    scores = prediction["scores"]
    for name, value in (("boxes", boxes), ("labels", labels), ("scores", scores)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Detection prediction field '{name}' for image {image_index} "
                f"must be a torch.Tensor, got {type(value).__name__}"
            )

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(
            f"Detection prediction 'boxes' for image {image_index} must have "
            f"shape [N, 4], got {tuple(boxes.shape)}"
        )
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError(
            f"Detection prediction 'labels' and 'scores' for image {image_index} "
            "must have shape [N]"
        )
    if boxes.shape[0] != labels.shape[0] or boxes.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Detection prediction fields for image {image_index} must have "
            "the same number of boxes, labels, and scores"
        )


def _decode_raw_detection_predictions(
    predictions: Mapping,
    batch_size: int,
    num_classes: Optional[int],
    score_threshold: float,
) -> list[dict[str, torch.Tensor]]:
    """Decode standard batched DETR-style ``pred_logits``/``pred_boxes`` outputs.

    The supported DETR and ViT detector ``infer`` methods return class
    probabilities and normalized ``cxcywh`` boxes in this structure. A final
    class channel is treated as no-object when the configured class count
    indicates that convention; models that have already removed that channel
    are decoded without an additional background filter.
    """
    missing = _RAW_DETECTION_KEYS.difference(predictions)
    if missing:
        raise ValueError(
            "Raw detection predictions must contain both 'pred_logits' and "
            f"'pred_boxes'; missing: {', '.join(sorted(missing))}"
        )

    logits = predictions["pred_logits"]
    boxes = predictions["pred_boxes"]
    if not isinstance(logits, torch.Tensor) or not isinstance(boxes, torch.Tensor):
        raise TypeError("Raw detection 'pred_logits' and 'pred_boxes' must be torch.Tensor values")
    if logits.ndim != 3:
        raise ValueError(
            f"Raw detection 'pred_logits' must have shape [B, Q, C], got {tuple(logits.shape)}"
        )
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(
            f"Raw detection 'pred_boxes' must have shape [B, Q, 4], got {tuple(boxes.shape)}"
        )
    if logits.shape[:2] != boxes.shape[:2]:
        raise ValueError(
            "Raw detection 'pred_logits' and 'pred_boxes' must agree on batch and query dimensions"
        )
    if logits.shape[0] != batch_size:
        raise ValueError(
            f"Raw detection output batch size {logits.shape[0]} does not match "
            f"input batch size {batch_size}"
        )
    if logits.shape[-1] < 1:
        raise ValueError("Raw detection 'pred_logits' must contain at least one class channel")
    if logits.device != boxes.device:
        raise ValueError("Raw detection 'pred_logits' and 'pred_boxes' must be on the same device")

    if num_classes is None:
        num_classes = logits.shape[-1]
    try:
        num_classes = int(num_classes)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Model num_classes must be a positive integer, got {num_classes!r}") from error
    if num_classes < 1:
        raise ValueError(f"Model num_classes must be a positive integer, got {num_classes}")

    if logits.shape[-1] == num_classes + 1:
        class_scores = logits[..., :num_classes]
        no_object_class = num_classes
    elif logits.shape[-1] == num_classes:
        class_scores = logits
        no_object_class = None
    else:
        raise ValueError(
            f"Raw detection class dimension {logits.shape[-1]} does not match "
            f"configured num_classes={num_classes} (with or without one no-object channel)"
        )

    threshold = float(score_threshold)
    decoded = []
    for image_index in range(batch_size):
        scores, labels = class_scores[image_index].max(dim=-1)
        keep_object = torch.ones_like(labels, dtype=torch.bool)
        if no_object_class is not None:
            keep_object = logits[image_index].argmax(dim=-1) != no_object_class

        boxes_xyxy, valid_boxes = sanitize_cxcywh(boxes[image_index].detach())
        labels = labels[valid_boxes]
        scores = scores[valid_boxes]
        keep_object = keep_object[valid_boxes]
        keep = keep_object
        if threshold > 0.0:
            keep = keep & (scores >= threshold)

        decoded.append({
            "boxes": boxes_xyxy[keep].detach(),
            "labels": labels[keep].detach().long(),
            "scores": scores[keep].detach().float(),
        })
    return decoded


class Inferer:
    """Class for running inference on a model over a dataset."""

    def __init__(
        self,
        model: BaseModel,
        data_loader: Optional[DataLoader] = None,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        """Initialize inferer.

        Args:
            model: Model to run inference on.
            data_loader: Data loader for inference.
            device: Device to run inference on.
        """
        self.model = model
        self.model.eval()
        self.data_loader = data_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _normalize_detection_predictions(
        self, predictions, batch_size: int, score_threshold: float
    ):
        """Return decoded detections per image when the model produced detections."""
        config = getattr(self.model, "config", None)
        num_classes = getattr(config, "num_classes", None)

        if isinstance(predictions, Mapping):
            raw_keys = _RAW_DETECTION_KEYS.intersection(predictions)
            if raw_keys:
                return _decode_raw_detection_predictions(
                    predictions,
                    batch_size=batch_size,
                    num_classes=num_classes,
                    score_threshold=score_threshold,
                ), True

            if _looks_like_decoded_prediction(predictions):
                _validate_decoded_prediction(predictions, image_index=0)
                if batch_size != 1:
                    raise ValueError(
                        "A single decoded detection dictionary can only represent "
                        f"a batch of one image, got batch size {batch_size}"
                    )
                return [dict(predictions)], True
            return predictions, False

        if isinstance(predictions, (list, tuple)) and any(
            _looks_like_decoded_prediction(prediction) for prediction in predictions
        ):
            if len(predictions) != batch_size:
                raise ValueError(
                    f"Decoded detection output contains {len(predictions)} per-image "
                    f"predictions for an input batch of {batch_size} images"
                )
            decoded = list(predictions)
            for image_index, prediction in enumerate(decoded):
                if not isinstance(prediction, Mapping):
                    raise TypeError(
                        f"Decoded detection prediction for image {image_index} must be a mapping, "
                        f"got {type(prediction).__name__}"
                    )
                _validate_decoded_prediction(prediction, image_index)
            return [dict(prediction) for prediction in decoded], True

        return predictions, False

    @torch.no_grad()
    def infer(self, **kwargs):
        """Run model inference over the complete data loader.

        Detection predictions are returned as a flat list with one decoded
        ``boxes``/``labels``/``scores`` dictionary per image. Non-detection
        outputs retain their model-provided batch structure.
        """
        if self.data_loader is None:
            raise ValueError("Data loader is required for evaluation. Provide data_loader during initialization.")

        score_threshold = kwargs.get("score_threshold")
        if score_threshold is not None:
            try:
                score_threshold = float(score_threshold)
            except (TypeError, ValueError) as error:
                raise ValueError("score_threshold must be a finite non-negative number") from error
            if not math.isfinite(score_threshold):
                raise ValueError("score_threshold must be finite")
            if score_threshold < 0.0:
                raise ValueError("score_threshold must be non-negative")

        # Keep model-owned defaults (for example RT-DETR's 0.3 threshold)
        # when the caller does not explicitly supply a threshold.
        inference_kwargs = dict(kwargs)
        if score_threshold is None:
            inference_kwargs.pop("score_threshold", None)
        else:
            inference_kwargs["score_threshold"] = score_threshold
        decode_threshold = 0.0 if score_threshold is None else score_threshold

        all_predictions = []
        all_targets = []
        self.model = self.model.to(self.device)
        assert self.model.get_device().type == self.device.type, (
            f"Error: Model is on '{self.model.get_device().type}', but device is '{self.device.type}'."
        )

        print(f"Running inference on {len(self.data_loader)} batches... (Device: {self.device.type})")
        for batch in tqdm(self.data_loader):
            images, targets = handle_raw_batch(batch, self.device)
            infer_outputs = self.model.infer(images, targets, **inference_kwargs)
            if not isinstance(infer_outputs, tuple):
                raise TypeError(
                    f"{type(self.model).__name__}.infer must return a tuple, got {type(infer_outputs).__name__}"
                )
            if len(infer_outputs) == 3:
                _, img_preds, targets = infer_outputs
            elif len(infer_outputs) == 2:
                img_preds, targets = infer_outputs
            else:
                raise ValueError(
                    f"{type(self.model).__name__}.infer must return two or three values, "
                    f"got {len(infer_outputs)}"
                )

            batch_size = int(images.shape[0]) if images.ndim > 0 else 1
            img_preds, is_detection = self._normalize_detection_predictions(
                img_preds, batch_size=batch_size, score_threshold=decode_threshold
            )
            img_preds = to_device(img_preds, torch.device("cpu"))
            if is_detection:
                all_predictions.extend(img_preds)
            else:
                all_predictions.append(img_preds)

            if targets is not None:
                targets = to_device(targets, torch.device("cpu"))
                if isinstance(targets, (list, tuple)):
                    all_targets.extend(targets)
                elif isinstance(targets, torch.Tensor) and targets.ndim > 0:
                    all_targets.extend(targets)
                else:
                    all_targets.append(targets)

        print("Done.")
        return all_predictions, all_targets
