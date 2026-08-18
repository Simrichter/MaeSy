from dataclasses import dataclass
from typing import Dict, Tuple, List

from torchvision.transforms.v2 import Transform

from _maesy_core.model.backbones.base_backbone import BaseBackbone, BaseConfig
import torch


@dataclass
class OnnxBackboneConfig(BaseConfig):
    """
    Config class for MobileNet backbones

    Args:
        :param version: MobileNet version ('mobilenetv2' is currently the only option)
        :param image_size: Input image size (assumed square)
        :param pretrained: Whether to use pre-trained weights
        :param feature_scales: Specify which feature scale levels to calculate and return during forward pass (following resnet naming scheme)
    """
    image_size: int = 224
    onnx_path: str = ""

class OnnxBackbone:
    """ONNX Backbone for feature extraction."""

    def __init__(self, config: OnnxBackboneConfig):
        super().__init__()
        self.config = config
        self.type = f"onnx_backbone_{self.config.onnx_path}"
        import onnxruntime as ort
        self.ort_session = ort.InferenceSession(self.config.onnx_path)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
            Forward pass through the ONNX backbone. Returns the output tensor.
        """
        # Convert torch tensor to numpy array
        x_np = x.detach().cpu().numpy()
        # Run inference
        outputs = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: x_np})
        # Convert output back to torch tensor
        return {"c6": torch.tensor(outputs[0], device=x.device, dtype=x.dtype)}

    def get_feature_dims(self) -> Dict[str, torch.Size]:
        """
            Return the feature dimensions of the backbone for each feature scale as a dict {scale: feature_dim}
        """
        return {"c6": torch.Size(self.ort_session.get_outputs()[0].shape)}

    def get_feature_channels(self) -> Tuple[int, ...]:
        """
            Return the number of channels for each feature scale as a tuple
        """
        return (self.ort_session.get_outputs()[0].shape[1],)

    def get_transforms(self) -> List[Transform]:
        """
            Returns a list of backbone-specific transforms that will be applied to the input tensor.
            Especially helpful with pretrained weights that expect certain normalizations
        """
        raise NotImplementedError("get_transforms")