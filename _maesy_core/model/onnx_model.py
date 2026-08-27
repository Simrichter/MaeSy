from dataclasses import dataclass

import torch

from _maesy_core.model.backbones.onnx_backbone import OnnxBackbone, OnnxBackboneConfig
from _maesy_core.model.base_model import BaseModel, BaseConfig
from _maesy_core.model.heads import DummyHead

@dataclass
class OnnxModelConfig(BaseConfig):
    onnx_model_path: str = ""
    type: str = "onnx"
    drop_layers: int = 0

class OnnxModel(BaseModel[OnnxModelConfig]):
    def __init__(self, config: OnnxModelConfig):
        super().__init__(config)
        self.is_trainable = False

        self.backbone = OnnxBackbone(OnnxBackboneConfig(onnx_path=self.config.onnx_model_path))
        self.head = DummyHead()

    def get_model_hash(self) -> str:
        """
            Returns a hash of the model's configuration and parameters.
            This can be used to uniquely identify the model for caching or versioning purposes.
        """
        import hashlib
        import json

        # Create a hash of the model's configuration and parameters
        config_str = json.dumps(self.config.__dict__, sort_keys=True)
        params_str = json.dumps({k: v.tolist() for k, v in self.state_dict().items()}, sort_keys=True)
        combined_str = config_str + params_str
        return f"{self.config.onnx_model_path.split('/')[-1].replace('.', '_')}_{hashlib.md5(combined_str.encode()).hexdigest()}"