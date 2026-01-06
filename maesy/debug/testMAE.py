from maesy.pretraining import MaskedAutoencoderViT
from maesy.model import ModelConfig
import torch
from torchvision.utils import save_image
from torchvision.io import read_image
import torch.nn.functional as F

config = ModelConfig(
    image_size=224,
    patch_size=16,
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    num_classes=80,
    num_queries=100
)


def random_masking(x: torch.Tensor, mask_ratio: float):
    """
        Args:
        x: [B, N, D] - input sequence
        mask_ratio: Ratio of patches to mask
    Returns:
        x_masked: [B, N * (1 - mask_ratio), D] - masked sequence
        mask: [B, N] - binary mask (0 is keep, 1 is remove)
        ids_restore: [B, N] - indices to restore original order
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    # Random shuffle
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # Generate binary mask: 0 is keep, 1 is remove
    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


model = MaskedAutoencoderViT(config)

#img = torch.randn((1, 3, 224, 224))
img = read_image("test_image.png")[None, :3, :, :]/255
img = F.interpolate(img, 224)
save_image(img, "img1.png")

patches = model.patchify(img)
x, mask, ids_restore = random_masking(patches, mask_ratio=0.8)

# ids_restore = torch.arange(0, 196, dtype = torch.int64).reshape(1, 196).flip(dims=[1])

mask_tokens = model.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

img_masked = model.unpatchify(x_)
save_image(img_masked, "img1_masked.png")
