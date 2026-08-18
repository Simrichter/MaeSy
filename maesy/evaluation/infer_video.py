import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from _maesy_core.inference.inferer import Inferer
from _maesy_core.dataset import UnlabeledDataset
from _maesy_core.model import MAEConfig

from PIL import Image

from _maesy_core.model.model_tools.model_factory import create_model


def write_video_from_imgs(images:list, output_path:str, fps:int=30):
    images = [to_pil_image(img[0]) for img in images]  # images, just convert it into PIL.Image obj
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    print("Saving")
    images[0].save(output_path/"vid.gif", save_all=True, optimize=False, append_images=images[1:], loop=0)
    print("Done")
    # images = [img for img in images]
    # video = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*'XVID'), 150, (224, 224))
    # for image in images:
    #     video.write(image)

def main():
    path = "/home/simon/Desktop/maesy-training/video/img_data"
    # load images from path, convert to list of pil images
    img_list = []
    for img_name in tqdm(sorted(os.listdir(path))):
        if img_name.endswith(".jpg") or img_name.endswith(".png"):
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            img_list.append(img)
    write_video_from_imgs(img_list, "/home/simon/Desktop/maesy-training/inference_results", fps=30)
if __name__ == "__main__":
    main()

def infer_video(args):
    mae_config = MAEConfig(
        image_size=224,
        patch_size=16,
        embed_dim=384,
        num_layers=8,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.1,
        decoder_embed_dim=384,
        decoder_num_layers=4
    )

    # Create MAE model
    print("Creating MAE model...")
    model = create_model("mae", mae_config) # MaskedAutoencoderViT(config=mae_config,)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transfs = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])

    dataset = UnlabeledDataset(
        args.imgpath,
        transforms=transfs,
        # use_first_n=30,
        filetype=".jpg"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False) # TODO: Add Model + Config into checkpoint!
    model.load_state_dict(checkpoint['model_state_dict'])
    inf = Inferer(model, dataloader, device)

    all_predictions, all_targets = inf.infer(mask_ratio=0.65, seed=42)

    # save_images(all_predictions, [*range(len(all_predictions))], args.output_path)
    write_video_from_imgs(all_predictions, args.output_path, fps=30)

    # Save predictions
    # filenames = [i for i in range(len(all_predictions))]
    # save_images(all_predictions, filenames, args.outputpath)