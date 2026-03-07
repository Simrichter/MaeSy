import os
from pathlib import Path
from typing import List

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes, save_image
from tqdm import tqdm

from maesy import ObjectDetectionDataset
from maesy.dataset.bounding_box import BoundingBox
from maesy.training.utils import collate_detection_fn, handle_raw_batch


def visualize_from_dataset(dataset: str, output_dir, device=torch.device("cpu")):
    """
        Draws bounding boxes from a Dataset that provides images and annotations in the format used for training (i.e., with 'boxes' and 'labels' in the target dictionaries).

        Args:
            :param dataset: Path to an object detection dataset
            :param output_dir: Folder in which the visualized images are saved
            :param device: Device to run visualization on (default: auto-detect CUDA if available, otherwise CPU)
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = ObjectDetectionDataset(dataset, transforms=None)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_detection_fn)

    for i, batch in enumerate(tqdm(dataloader)):
        images, targets = handle_raw_batch(batch, device)
        # print(targets)
        # images = batch["image"]  # [B, C, H, W]
        # targets = batch["target"]  # List of target dictionaries
        img = images[0]
        targ = targets[0]["boxes"]
        targ[:, (0, 2)] *= img.shape[2]  # Scale x coordinates to image width
        targ[:, (1, 3)] *= img.shape[1]  # Scale y coordinates to image height
        boxes = box_convert(targ, "cxcywh", "xyxy")
        img_with_boxes = draw_bounding_boxes(img, boxes)
        save_image(img_with_boxes, Path(output_dir)/f"{i}.png")

    print(f"Fertig! Annotierte Bilder liegen in: {output_dir}")

def visualize_annotations(input_dir, output_dir):
    """
        Zeichnet Bounding Boxen (im YOLO Format) in Bilder

        Args:
            :param input_dir: Ordner mit Bildern und Annotationen (.txt im YOLO Format)
            :param output_dir: Ordner, in dem die annotierten Bilder gespeichert werden
    """
    if output_dir!="" and os.path.exists(output_dir):
        raise ValueError(f"Failed: Output directory {output_dir} does not exist. Leave unspecified to create a 'visualized' subfolder in the input directory.")
    if output_dir=="":
        output_dir = os.path.join(input_dir, "visualized")
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        suffix = "."+file.split(".")[-1]
        if suffix in (".png", ".jpg", ".jpeg"):
            img_path = os.path.join(input_dir, file)
            txt_path = os.path.join(input_dir, file.replace(suffix, ".txt"))

            # Bild laden
            boxes = [BoundingBox.from_str(l) for l in open(txt_path, "r").readlines()] if os.path.exists(txt_path) else []
            if len(boxes) == 0:
                print(f"No boxes found for image {img_path}, skipping visualization.\n (Boxes: {boxes})")
                continue
            img = draw_boxes_in_image(img_path, boxes).float() / 255.0
            # Annotiertes Bild speichern
            out_path = os.path.join(output_dir, file)
            save_image(img, out_path)

    print(f"Fertig! Annotierte Bilder liegen in: {output_dir}")

def draw_boxes_in_image(img: str | torch.Tensor, boxes: List[BoundingBox] | torch.Tensor, labels: List[str] = None) -> torch.Tensor:
    """
        Draws bounding boxes from YOLO annotation file on the image.
    """
    color_coding = {
        0: (0, 0, 255),  # Rot für "soccer ball"
        1: (255, 0, 0),  # Blau für "Robot"
        2: (0, 255, 0)  # Grün für "LineCrossing"
    }
    name_coding = {
        0: "Ball",  # TODO: Get this stuff from model config?
        1: "Robot",
        2: "PenaltyCross",  # (Keine 27 Beschriftungen erwünscht), alternativ: LineCrossing
        3: "No-Object"
    }

    if type(img) is str:
        img = read_image(img)

    if labels is None:
        labels = [name_coding[box.cls()] for box in boxes]



    # Convert BoundingBox objects to tensor format if needed
    if type(boxes) is list:
        if len(boxes) == 0:
            return img
        boxes = torch.stack([box.coordinates_as_tensor()[0] for box in boxes])

    # Assuming normalized boxes:
    boxes[:, (0, 2)] *= img.shape[2]  # Scale x coordinates to image width
    boxes[:, (1, 3)] *= img.shape[1]  # Scale y coordinates to image height

    try:
        out = draw_bounding_boxes(img, boxes, labels=labels) #, colors=[*color_coding.values()]
    except ValueError as e:
        out = img
        print(f"Failed to draw boxes for image {img} with boxes {boxes} and labels {labels}\n\nError: {e}")

    return out