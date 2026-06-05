import os
import warnings
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL.ImageColor import getrgb
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes, save_image
from tqdm import tqdm

from maesy import ObjectDetectionDataset
from maesy.dataset import MaesyDataset
from maesy.dataset.bounding_box import BoundingBox
from maesy.training.utils import collate_detection_fn, handle_raw_batch


# def visualize_from_dataset(dataset: str, output_dir, device=torch.device("cpu")):
#     """
#         Draws bounding boxes from a Dataset that provides images and annotations in the format used for training (i.e., with 'boxes' and 'labels' in the target dictionaries).
#
#         Args:
#             :param dataset: Path to an object detection dataset
#             :param output_dir: Folder in which the visualized images are saved
#             :param device: Device to run visualization on (default: auto-detect CUDA if available, otherwise CPU)
#     """
#     if output_dir!="" and os.path.exists(output_dir):
#         raise ValueError(f"Failed: Output directory {output_dir} does not exist. Leave unspecified to create a 'visualized' subfolder in the input directory.")
#     if output_dir=="":
#         output_dir = os.path.join(dataset, "visualized")
#     os.makedirs(output_dir, exist_ok=True)
#     dataset = ObjectDetectionDataset(dataset, transforms=None)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_detection_fn)
#
#     for i, batch in enumerate(tqdm(dataloader)):
#         images, targets = handle_raw_batch(batch, device)
#         # print(targets)
#         # images = batch["image"]  # [B, C, H, W]
#         # targets = batch["target"]  # List of target dictionaries
#         img = images[0]
#         targ = targets[0]["boxes"]
#         targ[:, (0, 2)] *= img.shape[2]  # Scale x coordinates to image width
#         targ[:, (1, 3)] *= img.shape[1]  # Scale y coordinates to image height
#         boxes = box_convert(targ, "cxcywh", "xyxy")
#         img_with_boxes = draw_bounding_boxes(img, boxes)
#         save_image(img_with_boxes, Path(output_dir)/f"{i}.png")
#
#     print(f"Fertig! Annotierte Bilder liegen in: {output_dir}")

def visualize_dataset(dataset: MaesyDataset, output_dir: str, label_file: str = ""):
    """
        Visualizes bounding boxes and lines from a MaesyDataset.

        Args:
            :param dataset: The MaesyDataset to visualize
            :param output_dir: Output folder. !Expected to already exist!
            :param label_file: Path to a file that contains the class names in order (default: empty, i.e. use default class names from MaesyDataset yaml)
    """

    for idx, (img, targets) in tqdm(enumerate(dataset)):
        drawn = draw_objects_in_tensor((img*255).to(torch.uint8), targets["boxes"], targets["labels"][:len(targets["boxes"])], targets["line_points"], targets["ellipses"])
        out_path = os.path.join(output_dir, dataset.get_image_path(idx).name)
        save_image(drawn/255, out_path)
        # save_image(img, out_path)

def visualize_data(input_dir: str, output_dir: str, label_path:str= "", label_file:str= "", enable_lines: bool = True, enable_ellipses: bool = True, special_classes: Optional[dict[str, int]]=None):
    """
        Visualizes bounding boxes and lines. Autodetects MaesyDataset or standard image folder

        Args:
            :param input_dir: Root folder of MaesyDataset or path to images (autodetects)
            :param output_dir: Output folder to be created. Default: 'Visualized' subfolder in the input directory
            :param label_path: Path to a folder that contains the annotation text files (Leave empty, if MaesyDataset or annotations in input_dir)
            :param label_file: Path to a file that contains the class names in order (default: empty, i.e. use default class names / class names from MaesyDataset yaml)
            :param enable_lines: Whether to visualize lines
            :param enable_ellipses: Whether to visualize lines
            :param special_classes: Optional dict to specify the class IDs special classes (for example: {"ellipses": 3, "lines": 5}s
    """
    # TODO: Use MaesyDataset yaml for class names


    if output_dir!="" and os.path.exists(output_dir):
        raise ValueError(f"Failed: Output directory {output_dir} already exists. Leave unspecified to create a 'visualized' subfolder in the input directory.")
    if output_dir=="":
        output_dir = os.path.join(input_dir, "visualized")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Starting visualization...")
    print("=" * 60)

    datasets = []
    for split in ["train", "val", "test"]:
        try:
            datasets.append(MaesyDataset(input_dir, split, "detection", enable_lines=enable_lines, enable_ellipses=enable_ellipses))
        except AssertionError as e:
            ...
            print(e)
        except ValueError as e:
            ...
            print(e)
        except FileNotFoundError as e:
            ...
            print(e)
    if len(datasets) > 0:
        for d in datasets:
            visualize_dataset(d, output_dir)
        return
    else:
        print("No MaesyDataset detected, assuming standard image folder with annotations (txt files with same name as images).")

    # TODO: Line visualization not working!! Probably because line class id not known

    if label_file!="":
        with open(label_file, "r") as f:
            name_coding = {i: line.strip() for i, line in enumerate(f.readlines())}
        print(name_coding)
    else:
        name_coding = None

    if special_classes is None:
        special_classes = {}

    lbl_dir = input_dir if label_path=="" else Path(label_path)

    for file in os.listdir(input_dir):
        suffix = "."+file.split(".")[-1]
        if suffix in (".png", ".jpg", ".jpeg"):
            img_path = os.path.join(input_dir, file)
            txt_path = os.path.join(lbl_dir, file.replace(suffix, ".txt"))

            # Bild laden
            if not os.path.exists(txt_path):
                print(f"WARNING: No annotation file found for image {img_path}, skipping visualization.\n (Expected annotation file: {txt_path})")
                continue

            boxes = []
            labels = []
            lines = []
            ellipses = []
            for l in open(txt_path, "r").readlines():
                parts = l.split(" ")
                cls_id = int(parts[0])
                if cls_id not in special_classes.values():
                    boxes.append((box := BoundingBox.from_str(l, xyxy=True)).coordinates_as_tensor())
                    labels.append(box.cls())
                elif cls_id == special_classes["lines"]:
                    lines.append([float(x) for x in parts[1:]])
                elif cls_id == special_classes["ellipses"]:
                    ellipses.append([float(x) for x in parts[1:]])

                if len(boxes) == 0:
                    print(f"Empty annotations for image {img_path}, skipping visualization.\n (Boxes: {boxes})")
                    continue
                # img = draw_boxes_in_image(img_path, boxes, name_coding=name_coding).float() / 255.0
            img = read_image(img_path)[:3]  # Convert RGBA to RGB by dropping alpha channel if necessary
            if len(boxes)>0:
                b = torch.stack(boxes)
                img = _draw_boxes_in_tensor(img, b, labels=torch.tensor(labels), name_coding=name_coding, xyxy=True)
            img = _draw_lines_in_tensor(img, torch.tensor(lines))
            img = _draw_ellipses_in_tensor(img, torch.tensor(ellipses))
            img = img / 255.0

            # Annotiertes Bild speichern
            out_path = os.path.join(output_dir, file)
            save_image(img, out_path)

            # drawn = draw_objects_in_tensor((img * 255).to(torch.uint8), targets["boxes"], targets["labels"][:len(targets["boxes"])], targets["line_points"],
            #                                targets["ellipses"])
            # out_path = os.path.join(output_dir, dataset.get_image_path(idx).name)
            # save_image(drawn / 255, out_path)

    print(f"Fertig! Annotierte Bilder liegen in: {output_dir}")

# def draw_boxes_in_image(img: str | torch.Tensor, boxes: List[BoundingBox] | torch.Tensor, labels: List[str] = None, name_coding: dict[int, str]=None) -> torch.Tensor:
#     """
#         Draws bounding boxes from YOLO annotation file on the image.
#     """
#     colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple", "pink", "brown", "gray", "black"]
#     color_coding = {i: c for i, c in enumerate(colors)}
#
#     if name_coding is None:
#         # name_coding = {'CenterCircle': 9, 'CenterMark': 6, 'CornerArc': 10, 'FIFA 26 Ball': 0, 'GoalPost': 4, 'K1': 3, 'Lines': 8, 'Nao': 2, 'PenaltyMark': 5, 'Referee': 7, 'SPL Ball': 1}
#         name_coding = {'PenaltyCross': 2, 'Robot': 1, 'Ball': 0}
#
#         name_coding = {k: v for v, k in name_coding.items()}
#
#         # name_coding = {
#         #     0: "Ball",  # TODO: Get this stuff from model config?
#         #     1: "Robot",
#         #     2: "PenaltyCross",  # (Keine 27 Beschriftungen erwünscht), alternativ: LineCrossing
#         #     3: "No-Object"
#         # }
#
#     if type(img) is str:
#         img = read_image(img)
#         if img.shape[0] > 2:
#             img = img[:3]  # Convert RGBA to RGB by dropping alpha channel
#
#     if labels is None:
#         labels = [name_coding.get(box.cls(), "?") for box in boxes]
#         colors = [color_coding[box.cls()] for box in boxes]
#     else:
#         colors = ["red" if label == "Ball" else "blue" if label == "Robot" else "green" if label == "PenaltyCross" else "yellow" for label in labels]
#
#
#     # Convert BoundingBox objects to tensor format if needed
#     if type(boxes) is list:
#         if len(boxes) == 0:
#             return img
#         boxes = torch.stack([box.coordinates_as_tensor() for box in boxes])
#
#     # Assuming normalized boxes:
#     boxes[:, (0, 2)] *= img.shape[2]  # Scale x coordinates to image width
#     boxes[:, (1, 3)] *= img.shape[1]  # Scale y coordinates to image height
#
#     try:
#         out = draw_bounding_boxes(img, boxes, labels=labels, colors=colors)
#     except ValueError as e:
#         out = img
#         print(f"Failed to draw boxes for image of shape {img.shape} with boxes {boxes} and labels {labels}\n\nError: {e}")
#
#     return out

def _draw_boxes_in_tensor(img: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, name_coding: dict[int, str]=None, color_coding: dict[int, str]=None, xyxy: bool = False) -> torch.Tensor:
    """
        Draws bounding boxes on the image. If specified, uses labels and colors. If boxes is empty, img is returned

        Args:
            :param img: The image as a torch.Tensor in format [C, H, W] (pixel range: [0,255], dtype=torch.uint8)
            :param boxes: Tensor of bounding boxes in normalized xyxy format, shape [nb, 4]
            :param labels: Tensor of labels, shape [nb, ]
            :param name_coding: Optional dict to translate from class id to label name
            :param color_coding: Optional dict to translate from class id to color
            :param xyxy: Whether the boxes are already in xyxy format

        returns:
            img: The tensor with drawn bounding boxes, in format [C, H, W] (pixel range: [0,255])
    """
    if boxes is None or boxes.shape[0]==0:
        return img
    if not img.dtype == torch.uint8:
        warnings.warn("img does not have dtype uint8")
        img = img.to(torch.uint8)

    if color_coding is None:
        # colors = ["blue"]* len(boxes)
        colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple", "pink", "brown", "gray",
                  "black"]
        color_coding = {i: c for i, c in enumerate(colors)}
    # else:
    colors = [color_coding[cls_id.item()] for cls_id in labels]

    if name_coding is None:
        rendered_labels = [f"C{cls_id.item()}" for cls_id in labels]
    else:
        rendered_labels = [name_coding[cls_id.item()] for cls_id in labels]


    h, w = img.shape[-2:]
    boxes_xyxy = boxes.detach().cpu()
    boxes_xyxy[:, (0, 2)] *= w
    boxes_xyxy[:, (1, 3)] *= h
    boxes_xyxy[:, (0, 2)] = boxes_xyxy[:, (0, 2)].clamp(0, w - 1)
    boxes_xyxy[:, (1, 3)] = boxes_xyxy[:, (1, 3)].clamp(0, h - 1)
    # print(boxes_xyxy)
    drawn = draw_bounding_boxes(img, boxes_xyxy, labels=rendered_labels, colors=colors, width=1)
    return drawn

def _draw_lines_in_tensor(img: torch.Tensor, lines: torch.Tensor, line_color: str= "pink") -> torch.Tensor:
    """
    Draws lines in image tensor using cv2

    Args:
        :param img: The image as a torch.Tensor in format [C, H, W] (pixel range: [0,255], dtype=torch.uint8)
        :param lines: Tensor containing the lines as two keypoints in normalized xyxy format, shape [nl, 4]
        :param line_color: The color to draw the lines with. Default is pink.

    returns:
        img: The tensor with drawn bounding boxes, in format [C, H, W] (pixel range: [0,1])
    """
    drawn = img.detach().cpu().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    h, w = img.shape[-2:]
    for line in lines:
        x1, y1, x2, y2 = line.detach().cpu()
        x1 = int(x1.item() * w)
        y1 = int(y1.item() * h)
        x2 = int(x2.item() * w)
        y2 = int(y2.item() * h)
        cv2.line(drawn, (x1, y1), (x2, y2), color=getrgb(line_color), thickness=1)  # cv2.line draws in-place !!!
    drawn = torch.from_numpy(drawn).permute(2, 0, 1).contiguous().float()
    return drawn

def _draw_ellipses_in_tensor(img: torch.Tensor, ellipses: torch.Tensor, color: str="red") -> torch.Tensor:
    """
        Draws ellipses in image tensor using cv2

        Args:
            :param img: The image as a torch.Tensor in format [C, H, W] (pixel range: [0,255], dtype=torch.uint8)
            :param ellipses: Tensor containing the ellipses in normalized cx cy log_a log_b cos(2*theta) sin(2*theta) format, shape [ne, 6]
            :param color: The color to draw the lines with. Default is pink. !Expects

        returns:
            img: The tensor with drawn bounding boxes, in format [C, H, W] (pixel range: [0,1])
    """

    def _ellipse_to_points_cholesky(params, num=100):
        """
        Convert ellipse in cholesky representation to points in normalized xy format
        Args:
            :param params: [N, 5] = [N, (cx, cy, l11, l21, l22)]
            :param num: Number of points to Sample per ellipse
        returns: Tensor [N, num, 2]
        """
        cx, cy, l11, l21, l22 = params.unbind(-1)
        # build L: [N, 2, 2]
        zeros = torch.zeros_like(l11)
        L = torch.stack([
            torch.stack([l11, zeros], dim=-1),
            torch.stack([l21, l22], dim=-1)
        ], dim=-2)

        # unit circle
        t = torch.linspace(0, 2 * torch.pi, num, device=params.device, dtype=params.dtype)
        circle = torch.stack([torch.cos(t),torch.sin(t)], dim=0)  # [2, num]

        # expand for batch
        circle = circle.unsqueeze(0).expand(params.shape[0], -1, -1)  # [N, 2, num]

        # solve L x = circle  → x = L^{-1} circle
        pts = torch.linalg.solve(L.transpose(-1, -2), circle)  # [N, 2, num]

        # add center
        pts[:, 0, :] += cx.unsqueeze(-1)
        pts[:, 1, :] += cy.unsqueeze(-1)

        return pts.permute(0, 2, 1)  # [N, num, 2]

    def _ellipse_to_points(params, num=100):
        """
        Convert ellipse in cx, cy, a, b, theta representation to points in normalized xy format
        Args:
            :param params: [N, 5] = [N, (cx, cy, a, b, theta)]
            :param num: Number of points to Sample per ellipse
        returns: Tensor [N, num, 2]
        """
        cx, cy, log_a, log_b, cos2theta, sin2theta = params.unbind(-1)
        a, b = torch.exp(log_a), torch.exp(log_b)
        theta = torch.atan2(sin2theta, cos2theta) / 2

        # Parametric sampling of an ellipse with semi-axes a, b rotated by theta around center (cx, cy)
        # t: [num]
        t = torch.linspace(0, 2 * torch.pi, num, device=params.device, dtype=params.dtype)
        cos_t = torch.cos(t).unsqueeze(0)  # [1, num]
        sin_t = torch.sin(t).unsqueeze(0)  # [1, num]

        # local coordinates for each ellipse: [N, num]
        x_local = a.unsqueeze(-1) * cos_t
        y_local = b.unsqueeze(-1) * sin_t

        cos_theta = torch.cos(theta).unsqueeze(-1)  # [N, 1]
        sin_theta = torch.sin(theta).unsqueeze(-1)  # [N, 1]

        x = cx.unsqueeze(-1) + (x_local * cos_theta - y_local * sin_theta)
        y = cy.unsqueeze(-1) + (x_local * sin_theta + y_local * cos_theta)

        pts = torch.stack([x, y], dim=-1)  # [N, num, 2]
        return pts

    h, w = img.shape[-2:]
    if ellipses.shape[0] == 0:
        return img
    pts = _ellipse_to_points(ellipses, num=100).squeeze(0).numpy()
    pts[:, ::2] = pts[:, ::2] * w
    pts[:, 1::2] = pts[:, 1::2] * h
    pts = pts.reshape(-1, 1, 2).astype(np.int32)
    drawn = cv2.polylines(img.detach().cpu().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy(), [pts], isClosed=True, color=getrgb(color), thickness=2)
    drawn = torch.from_numpy(drawn).permute(2, 0, 1).contiguous().float()
    return drawn

def draw_objects_in_tensor(img: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor, lines: torch.Tensor, ellipses: torch.Tensor, name_coding: dict[int, str]=None, color_coding: dict[int, str]=None, xyxy: bool = False) -> torch.Tensor:
    """
        Draws bounding boxes and lines on the image.

        Args:
            :param img: Tensor containing the image in format [C, H, W] (pixel range: [0,255], dtype=torch.uint8)
            :param boxes: Tensor of bounding boxes in normalized xyxy format, shape [nb, 4]
            :param labels: Tensor of bbox labels, shape [nb, ]
            :param lines: Tensor of lines in normalized xyxy format, shape [nl, 4]
            :param ellipses: Tensor of ellipses in normalized cxcylll format, shape [ne, 5]
            :param name_coding: Optional dict to translate from class id to label name
            :param color_coding: Optional dict to translate from class id to color
            :param xyxy: Whether the boxes are already in xyxy format

    """
    img = _draw_boxes_in_tensor(img, boxes, labels, name_coding, color_coding, xyxy)
    img = _draw_lines_in_tensor(img, lines)
    img = _draw_ellipses_in_tensor(img, ellipses)
    return img
