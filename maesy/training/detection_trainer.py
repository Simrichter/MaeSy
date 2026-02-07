from typing import Optional

import torch

from ..model.components import Utils

from maesy.training import BaseTrainer

class DetectionTrainer(BaseTrainer):
    """Trainer for Masked Autoencoder Vision Transformer (MAE ViT) models.

    This trainer handles the training loop, loss computation, and optimization
    specific to MAE ViT models.
    """

    def forward_model(self, images: torch.Tensor, targets: Optional[torch.Tensor], val: bool) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            :param images: Input images [B, C, H, W].
            :param targets: Target images for reconstruction (are not used in MAE training, only there for consistency with superclass).
            :param val: Whether this is a validation pass

        Returns:
            model_out: The output from the model after postprocessing [B, C, H, W].
        """
        out, additional_data = self.model.forward(images, mask_ratio=0)
        losses = self.loss(out, targets)

        # if val:
        #     model_out = self.model.reconstruct(out, images, **additional_data)
        #     img_prediction = model_out[0] # torch.cat((model_out[0], images[0]), dim=-1)
        #     losses["img_out"] = img_prediction

        return losses