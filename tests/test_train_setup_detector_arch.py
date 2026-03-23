import pytest

from maesy.model import DETR, RTDETR
from maesy.training.train_setups.train_object_detection import _build_detection_model


def test_build_detection_model_supports_detr_and_rt_detr():
    detr_model = _build_detection_model("detr")
    rt_detr_model = _build_detection_model("rt_detr")

    assert isinstance(detr_model, DETR)
    assert isinstance(rt_detr_model, RTDETR)


def test_build_detection_model_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unsupported detector architecture"):
        _build_detection_model("unknown")

