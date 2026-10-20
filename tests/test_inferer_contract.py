from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from _maesy_core.inference.inferer import Inferer


class _InferModel(nn.Module):
    def __init__(self, return_count=2, output=None, num_classes=2):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(num_classes=num_classes)
        self.return_count = return_count
        self.output = _raw_predictions() if output is None else output
        self.seen_batch_sizes = []
        self.seen_thresholds = []
        self.seen_kwargs = []

    def get_device(self):
        return self.weight.device

    def infer(self, images, targets, **kwargs):
        self.seen_batch_sizes.append(images.shape[0])
        self.seen_thresholds.append(kwargs.get("score_threshold"))
        self.seen_kwargs.append(kwargs)
        predictions = self.output(images) if callable(self.output) else self.output
        if self.return_count == "not-a-tuple":
            return predictions
        if self.return_count == 0:
            return ()
        if self.return_count == 1:
            return (predictions,)
        if self.return_count == 2:
            return predictions, targets
        if self.return_count == 3:
            return {}, predictions, targets
        if self.return_count == 4:
            return {}, predictions, targets, None
        raise AssertionError(f"Unexpected return count {self.return_count}")


def _raw_predictions(batch_size=2):
    # DETR and ViT inference outputs contain foreground classes plus no-object.
    return {
        "pred_logits": torch.tensor(
            [
                [[0.80, 0.10, 0.10], [0.05, 0.05, 0.90]],
                [[0.05, 0.85, 0.10], [0.10, 0.10, 0.80]],
            ],
            dtype=torch.float32,
        )[:batch_size],
        # Boxes are normalized cxcywh and are decoded to xyxy for metrics.
        "pred_boxes": torch.tensor(
            [
                [[0.50, 0.50, 0.40, 0.20], [0.50, 0.50, 0.10, 0.10]],
                [[0.25, 0.50, 0.20, 0.40], [0.50, 0.50, 0.10, 0.10]],
            ],
            dtype=torch.float32,
        )[:batch_size],
    }


def _make_loader(sample_count=2, batch_size=2, collate_fn=None):
    samples = [
        (
            torch.full((3, 4, 4), float(index)),
            {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.long),
            },
        )
        for index in range(sample_count)
    ]

    if collate_fn is None:
        def collate_fn(batch):
            images, targets = zip(*batch)
            return torch.stack(images), list(targets)

    return DataLoader(samples, batch_size=batch_size, collate_fn=collate_fn)


@pytest.mark.parametrize("return_count", [2, 3])
def test_inferer_decodes_batched_raw_outputs_and_preserves_targets(return_count):
    model = _InferModel(return_count=return_count)
    inferer = Inferer(model, _make_loader(), device="cpu")

    predictions, targets = inferer.infer(score_threshold=0.5)

    assert model.seen_batch_sizes == [2]
    assert model.seen_thresholds == [0.5]
    assert len(predictions) == len(targets) == 2
    torch.testing.assert_close(predictions[0]["boxes"], torch.tensor([[0.3, 0.4, 0.7, 0.6]]))
    assert predictions[0]["labels"].tolist() == [0]
    assert predictions[0]["scores"].tolist() == pytest.approx([0.8])
    torch.testing.assert_close(predictions[1]["boxes"], torch.tensor([[0.15, 0.3, 0.35, 0.7]]))
    assert predictions[1]["labels"].tolist() == [1]
    assert predictions[1]["scores"].tolist() == pytest.approx([0.85])
    assert all(target["boxes"].shape == (0, 4) for target in targets)
    assert all(value.device.type == "cpu" for value in predictions[0].values())


def test_inferer_filters_background_and_invalid_boxes_without_losing_alignment():
    raw = {
        "pred_logits": torch.tensor(
            [[[0.70, 0.20, 0.10], [0.10, 0.80, 0.10], [0.10, 0.10, 0.80], [0.90, 0.05, 0.05]]]
        ),
        "pred_boxes": torch.tensor(
            [[[0.50, 0.50, 0.40, 0.20], [0.25, 0.50, 0.20, 0.40],
              [0.50, 0.50, 0.10, 0.10], [0.50, 0.50, -0.20, 0.20]]]
        ),
    }
    inferer = Inferer(
        _InferModel(output=raw),
        _make_loader(sample_count=1, batch_size=1),
        device="cpu",
    )

    predictions, _ = inferer.infer(score_threshold=0.0)

    assert predictions[0]["labels"].tolist() == [0, 1]
    assert predictions[0]["scores"].tolist() == pytest.approx([0.70, 0.80])
    torch.testing.assert_close(
        predictions[0]["boxes"],
        torch.tensor([[0.3, 0.4, 0.7, 0.6], [0.15, 0.3, 0.35, 0.7]]),
    )


def test_inferer_decodes_foreground_only_outputs_with_threshold():
    raw = {
        "pred_logits": torch.tensor([[[0.70, 0.30], [0.15, 0.85]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.4, 0.2], [0.5, 0.5, 0.2, 0.4]]]),
    }
    inferer = Inferer(
        _InferModel(output=raw),
        _make_loader(sample_count=1, batch_size=1),
        device="cpu",
    )

    predictions, _ = inferer.infer(score_threshold=0.8)

    assert predictions[0]["labels"].tolist() == [1]
    assert predictions[0]["scores"].tolist() == pytest.approx([0.85])
    torch.testing.assert_close(predictions[0]["boxes"], torch.tensor([[0.4, 0.3, 0.6, 0.7]]))


def test_inferer_preserves_decoded_rt_detr_fields_and_flattens_per_image():
    decoded = [
        {
            "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
            "line_points": torch.tensor([[0.2, 0.3, 0.8, 0.3]]),
        },
        {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.long),
            "scores": torch.empty((0,)),
            "line_points": torch.empty((0, 4)),
        },
    ]
    model = _InferModel(return_count=3, output=decoded)
    inferer = Inferer(model, _make_loader(), device="cpu")

    predictions, _ = inferer.infer()

    assert model.seen_thresholds == [None]
    assert len(predictions) == 2
    assert predictions[0]["boxes"].tolist() == [[0.1, 0.2, 0.3, 0.4]]
    assert predictions[0]["line_points"].tolist() == [[0.2, 0.3, 0.8, 0.3]]
    assert predictions[1]["boxes"].shape == (0, 4)


@pytest.mark.parametrize(
    ("return_count", "output", "error_type", "message"),
    [
        ("not-a-tuple", _raw_predictions(), TypeError, "must return a tuple"),
        (0, _raw_predictions(), ValueError, "two or three values"),
        (1, _raw_predictions(), ValueError, "two or three values"),
        (4, _raw_predictions(), ValueError, "two or three values"),
        (2, {"pred_logits": torch.zeros((2, 2, 3))}, ValueError, "missing: pred_boxes"),
        (2, _raw_predictions(batch_size=1), ValueError, "does not match input batch size"),
        (
            2,
            {"pred_logits": torch.zeros((2, 2, 3)), "pred_boxes": torch.zeros((2, 1, 4))},
            ValueError,
            "agree on batch and query dimensions",
        ),
        (
            2,
            {"pred_logits": torch.zeros((2, 2, 4)), "pred_boxes": torch.zeros((2, 2, 4))},
            ValueError,
            "does not match configured num_classes",
        ),
        (
            2,
            {"pred_logits": [[[0.1, 0.8, 0.1]]], "pred_boxes": [[[0.5, 0.5, 0.2, 0.2]]]},
            TypeError,
            "must be torch.Tensor values",
        ),
    ],
)
def test_inferer_rejects_malformed_inference_contracts(return_count, output, error_type, message):
    inferer = Inferer(_InferModel(return_count=return_count, output=output), _make_loader(), device="cpu")

    with pytest.raises(error_type, match=message):
        inferer.infer()


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        (torch.tensor([[0]]), torch.tensor([0.9])),
        (torch.tensor([0]), torch.tensor([[0.9]])),
    ],
)
def test_inferer_rejects_invalid_decoded_field_shapes(labels, scores):
    decoded = {
        "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "labels": labels,
        "scores": scores,
    }
    inferer = Inferer(
        _InferModel(output=[decoded, decoded]),
        _make_loader(),
        device="cpu",
    )

    with pytest.raises(ValueError, match=r"must have shape \[N\]"):
        inferer.infer()


def test_inferer_rejects_single_decoded_mapping_for_multi_image_batch():
    decoded = {
        "boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "labels": torch.tensor([1]),
        "scores": torch.tensor([0.9]),
    }
    inferer = Inferer(_InferModel(output=decoded), _make_loader(), device="cpu")

    with pytest.raises(ValueError, match="batch of one image"):
        inferer.infer()


def test_inferer_preserves_nondetection_outputs_and_flattens_classification_targets():
    output = torch.zeros((2, 3))
    model = _InferModel(output=output)

    def classification_collate(batch):
        images, labels = zip(*batch)
        return torch.stack(images), torch.tensor(labels, dtype=torch.long)

    loader = DataLoader(
        [(torch.zeros((3, 4, 4)), 0), (torch.ones((3, 4, 4)), 1)],
        batch_size=2,
        collate_fn=classification_collate,
    )
    predictions, targets = Inferer(model, loader, device="cpu").infer()

    assert len(predictions) == 1
    torch.testing.assert_close(predictions[0], output)
    assert targets == [0, 1]


def test_inferer_collects_predictions_in_order_across_loader_batches():
    def make_predictions(images):
        centers_x = images[:, 0, 0, 0].unsqueeze(1) / 10 + 0.5
        logits = torch.tensor([0.9, 0.1, 0.0]).view(1, 1, 3).expand(images.shape[0], 1, 3).clone()
        boxes = torch.tensor([0.5, 0.5, 0.2, 0.2]).view(1, 1, 4).expand(images.shape[0], 1, 4).clone()
        boxes[:, :, 0] = centers_x
        return {"pred_logits": logits, "pred_boxes": boxes}

    model = _InferModel(output=make_predictions)
    inferer = Inferer(
        model,
        _make_loader(sample_count=3, batch_size=2),
        device="cpu",
    )

    predictions, targets = inferer.infer(score_threshold=0.0)

    assert model.seen_batch_sizes == [2, 1]
    assert len(predictions) == len(targets) == 3
    assert [target["labels"].numel() for target in targets] == [0, 0, 0]
    assert [prediction["labels"].tolist() for prediction in predictions] == [[0], [0], [0]]
    centers_x = [
        ((prediction["boxes"][:, 0] + prediction["boxes"][:, 2]) / 2).item()
        for prediction in predictions
    ]
    assert centers_x == pytest.approx([0.5, 0.6, 0.7])
    assert [prediction["scores"].item() for prediction in predictions] == pytest.approx([0.9, 0.9, 0.9])


@pytest.mark.parametrize("explicit_none", [False, True])
def test_inferer_omits_threshold_kwarg_when_using_model_default(explicit_none):
    model = _InferModel(return_count=3, output=[
        {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.long), "scores": torch.empty((0,))},
        {"boxes": torch.empty((0, 4)), "labels": torch.empty((0,), dtype=torch.long), "scores": torch.empty((0,))},
    ])

    options = {"score_threshold": None} if explicit_none else {}
    Inferer(model, _make_loader(), device="cpu").infer(**options)

    assert "score_threshold" not in model.seen_kwargs[0]
    assert model.seen_thresholds == [None]


def test_inferer_propagates_classification_and_other_model_kwargs():
    model = _InferModel(output=torch.zeros((2, 3)))
    Inferer(model, _make_loader(), device="cpu").infer(score_threshold=0.25, return_embeddings=True)
    assert model.seen_thresholds == [0.25]
    assert model.seen_kwargs[0]["return_embeddings"] is True
    assert model.seen_kwargs[0]["return_embeddings"] is True


@pytest.mark.parametrize("score_threshold", [float("nan"), float("inf"), -0.1, "invalid"])
def test_inferer_rejects_invalid_score_threshold(score_threshold):
    inferer = Inferer(_InferModel(), _make_loader(), device="cpu")

    with pytest.raises(ValueError, match="score_threshold"):
        inferer.infer(score_threshold=score_threshold)
