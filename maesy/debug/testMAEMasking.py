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
    num_layers=1,
    num_heads=1,
    num_classes=1,
    num_queries=1
)

def create_test_image():
    from PIL import Image, ImageDraw, ImageFont
    # Bildparameter
    img_size = config.image_size
    cell_size = config.patch_size
    grid_size = img_size // cell_size

    # Bild erstellen
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    # Schriftart laden
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except IOError:
        font = ImageFont.load_default()

    number = 1

    for row in range(grid_size):
        for col in range(grid_size):
            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            draw.rectangle([x0, y0, x1, y1], outline="black")
            text = str(number)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x0 + (cell_size - text_width) / 2
            text_y = y0 + (cell_size - text_height) / 2
            draw.text((text_x, text_y), text, fill="black", font=font)
            number += 1
    img.save("test_grid.png")
    return "test_grid.png"


model = MaskedAutoencoderViT(config, decoder_embed_dim=config.patch_size**2*3)

# create_test_image()
img = read_image("test_grid.png")[None, :3, :, :]/255
img = F.interpolate(img, config.image_size)

patches = model.patchify(img)
x, mask, ids_restore = model.random_masking(patches, mask_ratio=0.8)

mask_tokens = model.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  #1 # no cls token
x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

img_masked = model.unpatchify(x_)
save_image(img_masked, "img1_masked.png")