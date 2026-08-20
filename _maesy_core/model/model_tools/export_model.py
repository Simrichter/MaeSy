from pathlib import Path
import torch

from _maesy_core.model import MAEConfig
from _maesy_core.model.model_tools.model_factory import create_model_from_config

def export_model():
        mae_config = MAEConfig(
                image_size=224,
                patch_size=16,
                embed_dim=384,
                num_layers=8,
                num_heads=6,
                mlp_ratio=4.0,
                dropout=0.1,
                attention_dropout=0.1,
                decoder_embed_dim=256,
                decoder_num_layers=4
        )

        # Create MAE model
        print("Creating model...")
        model = create_model_from_config("ViTDetector", mae_config)# TransformerDetectionModel(mae_config)
        model.eval()

        # Create example inputs for exporting the model. The inputs should be a tuple of tensors.
        example_inputs = (torch.randn(1, 196,768),) # torch.arange(0, 196).unsqueeze(0)
        onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)

        path = Path("onnx")
        path.mkdir(parents=True, exist_ok=True)
        onnx_program.save(path/"image_classifier_model.onnx")

if  __name__ == "__main__":
        export_model()