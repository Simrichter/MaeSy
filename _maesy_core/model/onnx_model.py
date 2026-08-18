from dataclasses import dataclass

import torch

from _maesy_core.model.backbones.onnx_backbone import OnnxBackbone, OnnxBackboneConfig
from _maesy_core.model.base_model import BaseModel
from _maesy_core.model.heads import DummyHead

@dataclass
class OnnxModelConfig:
    onnx_model_path: str
    type: str = "onnx"
    drop_layers: int = 0

class OnnxModel(BaseModel):
    def __init__(self, config: OnnxModelConfig):
        super().__init__()
        self.config = config
        self.is_trainable = False

        self.backbone = OnnxBackbone(OnnxBackboneConfig(onnx_path=self.config.onnx_model_path))
        self.head = DummyHead()