from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import onnxruntime as ort
import onnx

from torchvision.transforms.v2 import Transform
import torch

from _maesy_core.model.backbones import BaseBackbone, BaseBackboneConfig


@dataclass
class OnnxBackboneConfig(BaseBackboneConfig):
    image_size: Tuple[int, int] = (224, 224)
    onnx_path: str = ""
    type: str = f"onnx_backbone_{onnx_path}"
    intermediate_output: Optional[str] = None

class OnnxBackbone(BaseBackbone):
    """ONNX Backbone for feature extraction."""

    def __init__(self, config: OnnxBackboneConfig):
        super().__init__()
        self.config:OnnxBackboneConfig = config
        if self.config.intermediate_output is not None:
            self._add_output_at_layer(self.config.intermediate_output)
        self._instantiate_session()

    def _instantiate_session(self):
        self.ort_session = ort.InferenceSession(self.config.onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def get_device(self) -> torch.device:
        """
            Returns the device on which the ONNX model is running.
        """
        providers = self.ort_session.get_providers()
        if 'CUDAExecutionProvider' in providers:
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def _add_output_at_layer(self, layer_name: str):
        """
            Add an output at a specific layer in the ONNX model.
            This allows for extracting features from intermediate layers.
            ! Manual re-instantiation of ORT Sessions is required after calling this method !
        """
        value_info_protos = []
        model = onnx.load(self.config.onnx_path)
        shape_info = onnx.shape_inference.infer_shapes(model)
        for idx, node in enumerate(shape_info.graph.value_info):
            if node.name == layer_name:
                value_info_protos.append(node)
        assert not len(value_info_protos) < 1, f"Layer {layer_name} not found in the model."
        assert not len(value_info_protos) > 1, f"Layer {layer_name} found multiple times ({len(value_info_protos)}) in the model."
        model.graph.output.extend(value_info_protos)  # in inference stage, these tensor will be added to output dict.
        onnx.checker.check_model(model)
        modded_path = self.config.onnx_path.removesuffix(".onnx") + "_clusterdevil_modified.onnx"
        onnx.save(model, modded_path)
        self.config.onnx_path = modded_path

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
            Forward pass through the ONNX backbone. Returns the output tensor.
        """
        if x.ndim == 4:
            assert x.shape[0] == 1, "Batch size must be 1 for ONNX inference."
            x = x.squeeze(0)  # Remove batch dimension for ONNX inference
        elif x.ndim == 3:
            pass  # Already in the correct shape
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, but got {x.ndim}D.")
        assert x.shape == self.ort_session.get_inputs()[0].shape, f"Input shape {x.shape} does not match expected shape {self.ort_session.get_inputs()[0].shape}"
        # Convert torch tensor to numpy array
        x_np = x.detach().cpu().numpy()
        # Run inference
        self.ort_session.get_outputs()
        outputs = self.ort_session.run(self.config.intermediate_output, {self.ort_session.get_inputs()[0].name: x_np})
        # Convert output back to torch tensor
        return {"output": torch.tensor(outputs[0], device="cpu", dtype=x.dtype)}

    def get_input_dims(self) -> torch.Size:
        """
            Return the input dimensions of the backbone as a torch.Size object
        """
        return torch.Size(self.ort_session.get_inputs()[0].shape)

    def get_feature_dims(self) -> Dict[str, torch.Size]:
        """
            Return the feature dimensions of the backbone for each feature scale as a dict {scale: feature_dim}
        """
        return {"output": torch.Size(self.ort_session.get_outputs()[0].shape)}

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