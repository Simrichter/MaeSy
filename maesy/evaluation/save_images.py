import os
from PIL import Image
import torch
import torchvision.transforms as T

def save_images(images, filenames, output_dir):
    """Save images to the specified output directory.

    Args:
        images: A batch of images as a tensor or a list of PIL Images.
        filenames: A list of filenames corresponding to each image.
        output_dir: Directory where images will be saved.
    """

    os.makedirs(output_dir, exist_ok=True)

    if isinstance(images, torch.Tensor):
        images = images.cpu()
        to_pil = T.ToPILImage()
        for img_tensor, filename in zip(images, filenames):
            img = to_pil(img_tensor)
            img.save(os.path.join(output_dir, filename))
    elif isinstance(images, list):
        if all(isinstance(img, torch.Tensor) for img in images):
            images = [img.cpu() for img in images]
            to_pil = T.ToPILImage()
            for img_tensor, filename in zip(images, filenames):
                img = to_pil(img_tensor[0])
                img.save(os.path.join(output_dir, str(filename)+".png"))
    else:
        for img, filename in zip(images, filenames):
            img.save(os.path.join(output_dir, filename))