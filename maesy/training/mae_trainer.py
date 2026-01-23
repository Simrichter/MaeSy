from typing import Optional

import torch

from ..model.components import Utils

from maesy.training import BaseTrainer

class MaeTrainer(BaseTrainer):
    """Trainer for Masked Autoencoder Vision Transformer (MAE ViT) models.

    This trainer handles the training loop, loss computation, and optimization
    specific to MAE ViT models.
    """

    def forward_model(self, images: torch.Tensor, targets: Optional[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            images: Input images [B, C, H, W].
            targets: Target images for reconstruction (are not used in MAE training, only there for consistency with superclass).

        Returns:
            model_out: The output from the model after postprocessing [B, C, H, W].
        """
        x = Utils.patchify(images, self.model.config.image_size, self.model.config.patch_size)
        x, mask, ids_shuffle = Utils.random_masking(x, self.config.mask_ratio)
        # print(x.shape, mask.shape, ids_shuffle.shape)
        out = self.model(x, **{"mask": mask, "ids_shuffle": ids_shuffle.to(device=self.device, non_blocking=True)})

        target = Utils.patchify(images, self.model.config.image_size, self.model.config.patch_size)
        # TODO: Remove ugly quick fix with patchify (maybe unpatchify mask?)
        # losses = self.loss(model_out, images, mask)
        losses = self.loss(out, target, mask)
        model_out = Utils.unpatchify(out.detach()*mask.unsqueeze(-1), self.model.config.image_size, self.model.config.patch_size)
        img_prediction = torch.cat((model_out[0], images[0]), dim=-1)

        losses["img_out"] = img_prediction
        return losses