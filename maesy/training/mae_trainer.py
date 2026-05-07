from typing import Optional

import torch

from ..model.components import Utils

from maesy.training import BaseTrainer

class MaeTrainer(BaseTrainer):
    """Trainer for Masked Autoencoder Vision Transformer (MAE ViT) models.

    This trainer handles the training loop, loss computation, and optimization
    specific to MAE ViT models.
    """

    @staticmethod
    def _sample_random_mask(batch_size: int, num_tokens: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
        len_keep = int(num_tokens * (1 - mask_ratio))
        noise = torch.rand(batch_size, num_tokens, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.ones([batch_size, num_tokens], device=device)
        mask[:, :len_keep] = 0
        return torch.gather(mask, dim=1, index=ids_restore)

    def _create_patch_mask(self, images: torch.Tensor) -> torch.Tensor:
        model_patch_size: int = self.model.config.patch_size
        mask_patch_size: int = getattr(self.config, "mask_patch_size", model_patch_size)
    #
        if mask_patch_size < model_patch_size or mask_patch_size % model_patch_size != 0:
            raise ValueError(
                f"mask_patch_size ({mask_patch_size}) must be a multiple of model patch_size ({model_patch_size})"
            )
    #
    #     patches_per_side = self.model.config.image_size // model_patch_size
    #     group = mask_patch_size // model_patch_size
    #     if patches_per_side % group != 0:
    #         raise ValueError(
    #             f"image_size/patch_size ({patches_per_side}) must be divisible by mask group size ({group})"
    #         )
    #
    #     coarse_side = patches_per_side // group
    #     coarse_mask = self._sample_random_mask(
    #         batch_size=images.shape[0],
    #         num_tokens=coarse_side * coarse_side,
    #         mask_ratio=self.config.mask_ratio,
    #         device=images.device,
    #     )
    #     coarse_mask = coarse_mask.view(images.shape[0], coarse_side, coarse_side)
    #     fine_mask = coarse_mask.repeat_interleave(group, dim=1).repeat_interleave(group, dim=2)
    #     return fine_mask.reshape(images.shape[0], -1)
        # model_patch_size = 4 # self.model.config.patch_size
        # mask_patch_size = 16 # getattr(self.config, "mask_patch_size", model_patch_size)

        image_size = images.shape[-1]
        patches_per_side = image_size // model_patch_size
        group = mask_patch_size // model_patch_size
        if patches_per_side % group != 0:
            raise ValueError(
                f"image_size/patch_size ({patches_per_side}) must be divisible by mask group size ({group})"
            )

        coarse_side = patches_per_side // group
        coarse_mask = self._sample_random_mask(
            batch_size=images.shape[0],
            num_tokens=coarse_side * coarse_side,
            mask_ratio=0.75,
            device=images.device,
        )
        coarse_mask = coarse_mask.view(images.shape[0], coarse_side, coarse_side)
        fine_mask = coarse_mask.repeat_interleave(group, dim=1).repeat_interleave(group, dim=2)  # .reshape(images.shape[0], -1)
        pixel_mask = fine_mask.repeat_interleave(model_patch_size, dim=1).repeat_interleave(model_patch_size, dim=2)
        return pixel_mask

    def _apply_mask_to_images(self, images: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = images.shape
        p = self.model.config.patch_size
        patches_per_side = self.model.config.image_size // p
        mask_2d = patch_mask.view(b, 1, patches_per_side, patches_per_side)
        pixel_mask = mask_2d.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        return images * (1.0 - pixel_mask)

    def forward_model(self, images: torch.Tensor, targets: Optional[torch.Tensor], val: bool) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            :param images: Input images [B, C, H, W].
            :param targets: Target images for reconstruction (are not used in MAE training, only there for consistency with superclass) [B, C, H, W].
            :param val: Whether this is a validation pass (outputs additional data for visualization if True).

        Returns:
            model_out: The output from the model after postprocessing [B, C, H, W].
        """
        pixel_mask = self._create_patch_mask(images)
        pixel_mask = pixel_mask.reshape(images.shape[0], *images.shape[2:]).unsqueeze(1).repeat(1, 3, 1, 1)
        masked_images = images * (1.0 - pixel_mask)
        out = self.model.forward(masked_images)
        additional_data = {"mask": pixel_mask, "masked_images": masked_images}

        losses = self.loss(out, images, additional_data["mask"])

        if val:
            img_prediction = out[0].detach() # torch.cat((model_out[0], images[0]), dim=-1)
            tgt_img = images[0].detach()
            img_prediction = (img_prediction * self._IMAGENET_STD) + self._IMAGENET_MEAN
            tgt_img = (tgt_img * self._IMAGENET_STD) + self._IMAGENET_MEAN

            img_prediction = img_prediction * pixel_mask[0] + tgt_img * (1.0 - pixel_mask[0])  # Mask out the known squares (since there is no training signal anyways)
            img_prediction = img_prediction.clamp(0.0, 1.0)
            losses["img_out"] = torch.cat((img_prediction, tgt_img), dim=-1).cpu() #.to(torch.uint8).cpu()


        return losses