"""Object detection dataset implementation."""

import os
from pathlib import Path

import torch
import torchvision.tv_tensors
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Dict, List, Tuple
import numpy as np
import yaml
import math

from maesy.dataset.bounding_box import BoundingBox


class MaesyDataset(Dataset):
    """Dataset for object detection in COCO format."""

    def __init__(
            self,
            dataset_dir: str,
            split: str,
            annotation_type: str,
            transforms: Optional[Callable] = None,
            start_index: int = 0,
            step: int = 1,
            repeat_factor: int = 1,
            enable_lines=False,
            enable_ellipses=False,
            use_first_n: int = None
    ):
        """
        Initialize ObjectDetectionDataset.

        Args:
            :param dataset_dir: Path to the MaesyDataset root directory
            :param split: The split to be used (i.e. "train", "val", "test", etc.)
            :param annotation_type: The type of dataset, choice of ["detection", "classification", "None"]
            :param transforms: Optional transforms to apply
            :param start_index: The index from which to start
            :param step: The step size for sampling images (e.g., step=2 will take every other image)
            :param repeat_factor: The factor by which to repeat the dataset (e.g., repeat_factor=2 will repeat the dataset twice, effectively doubling its size)
            :param enable_lines: Whether to include line annotations (if line_class_id is defined in dataset.yaml)
            :param enable_ellipses: Whether to include ellipse annotations (if ellipse_class_id is defined in dataset.yaml)
            :param use_first_n: If not None, only use the first n samples from the dataset (after applying start_index and step)
        """

        if not os.path.exists(dataset_dir):
            raise ValueError(f"Path '{dataset_dir}' does not exist.")

        if dataset_dir.endswith((".yaml", ".yml")):
            yaml_path = dataset_dir
        else:
            yaml_path = None
            yaml_candidates = [c for c in os.listdir(dataset_dir) if c.endswith((".yaml", ".yml"))]
            if len(yaml_candidates) < 1:
                raise ValueError(f"No dataset.yaml file found in dataset directory {dataset_dir}")
            elif len(yaml_candidates) > 1:
                raise ValueError(f"Multiple dataset.yaml files found in dataset directory {dataset_dir}\nFound yaml files: {yaml_candidates}\nSpecify the correct yaml directly as the dataset path")
            else:
                yaml_path = f"{dataset_dir}/{yaml_candidates[0]}"

        # Try to read dataset configuration file
        if yaml_path is not None and os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                yaml_data = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            yaml_data = {}

        dataset_dir = yaml_data.get("path", dataset_dir)
        self.box_format = str(yaml_data.get("box_format", "")).lower()
        if self.box_format not in {"xyxy", "cxcywh"}:
            raise ValueError(f"Unsupported box_format '{self.box_format}' in dataset.yaml. Expected 'xyxy' or 'cxcywh'.")

        split_path = Path(dataset_dir) / yaml_data.get(split, split)
        assert split_path.exists(), f"Requested split '{split}' does not exist in dataset at {dataset_dir}"

        self.images_dir = split_path / "images"
        self.annotations_dir = split_path / "labels"
        assert self.images_dir.exists()
        assert self.annotations_dir.exists()
        self.transforms = transforms
        self.return_labels = annotation_type != "None"

        self.id_to_name = {i:v for i,v in enumerate(yaml_data.get("names", []))}
        self.name_to_id = {v:k for k,v in self.id_to_name.items()}

        self.images: List[Path] = [Path(img) for img in sorted(os.listdir(self.images_dir)) if img.endswith((".jpg", ".jpeg", ".png"))][start_index::step] * repeat_factor

        if self.return_labels:
            self.num_classes = yaml_data.get("nc", -1)
            if self.num_classes == -1 and len(self.name_to_id) > 0:
                raise ValueError(f"Value of nc not set properly in dataset.yaml. Found {len(self.name_to_id)} classes in dataset.yaml, but no 'nc' entry")
            elif self.num_classes != -1 and self.num_classes != len(self.name_to_id):
                raise ValueError(f"Mismatch in dataset.yaml\nnc: {self.num_classes}, but {len(self.name_to_id)} class names were found.")

            self.special_classes = {'line_class_id': self.name_to_id.get(yaml_data.get("lines", None), -1),
                                    'ellipse_class_id': self.name_to_id.get(yaml_data.get("ellipses", None), -1)}
            self.enable_lines = enable_lines
            self.enable_ellipses = enable_ellipses

            self.annotations = []
            for img in self.images:
                annotation_path = Path(img).with_suffix(".txt")
                if not (self.annotations_dir / annotation_path).exists():
                    raise FileNotFoundError(f"Annotation file {annotation_path} not found for image {img}")
                self.annotations.append(annotation_path)

        # self.annotations: List[Path] = [Path(ann) for ann in sorted(os.listdir(self.annotations_dir)) if ann.endswith(".txt")][start_index::step]*repeat_factor
        if use_first_n is not None:
            self.images = self.images[:use_first_n]
            self.annotations = self.annotations[:use_first_n]

        print(f"Loaded {split} data...")
        print("-" * 30)

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[
        torchvision.tv_tensors.Image, List[Dict[str, torchvision.tv_tensors.BoundingBoxes]]]:
        """
        Get image and annotations at index.
        If transforms are provided, they are applied to the image before returning it.
        Otherwise, the image is returned as a tensor.
        Labels are returned as a list of BoundingBox instances.

        Args:
            :param idx: Index

        Returns:
            Dictionary containing the image and target annotations as a List[Object]
        """
        image_path = os.path.join(self.images_dir, self.images[idx])
        target = None
        # Load image
        with Image.open(image_path).convert('RGB') as image:
            if self.return_labels:
                img_width, img_height = image.size
                annotation_path = os.path.join(self.annotations_dir, self.annotations[idx])
                if annotation_path.split("/")[-1].split(".")[0] != image_path.split("/")[-1].split(".")[0]:
                    print("\n\nWARNING: Annotation file name does not match image file name! Check that the annotation file names in the labels folder match the image file names in the images folder (except for the extension). Annotation file: {}, Image file: {}\n\n".format(annotation_path, image_path))
                with open(annotation_path, "r") as f:
                    boxes_list: List[BoundingBox] = []
                    line_points_list: List[List[float]] = []
                    ellipse_points_list: List[List[float]] = []
                    for raw_line in f.readlines():
                        splits = raw_line.split()
                        cls_id = int(splits[0])
                        if self.special_classes['line_class_id'] is not None and cls_id == self.special_classes['line_class_id']:
                            if self.enable_lines:
                                assert len(splits) == 5, f"Invalid annotation format in {annotation_path}: '{raw_line.strip()}'. Expected 5 columns for annotation type 'line': 'class x1 y1 x2 y2'."
                                line_points_list.append([*map(float, splits[1:])])
                        elif self.special_classes['ellipse_class_id'] is not None and cls_id == self.special_classes['ellipse_class_id']:
                            if self.enable_ellipses:
                                assert len(splits) == 7, f"Invalid annotation format in {annotation_path}: '{raw_line.strip()}'. Expected 7 columns for annotation type 'ellipse': 'class center_x center_y log_a log_b cos(2*theta) sin(2*theta)'."
                                ellipse_points_list.append([*map(float, splits[1:])])
                        else:
                            expected = "class x1 y1 x2 y2" if self.box_format == "xyxy" else "class cx cy w h"
                            assert len(splits) == 5, (
                                f"Invalid annotation format in {annotation_path}: '{raw_line.strip()}'. "
                                f"Expected 5 columns for annotation type 'BoundingBox': '{expected}'."
                            )
                            if self.box_format == "cxcywh":
                                box = BoundingBox.from_cxcywh(cls_id, *map(float, splits[1:]), normalized=True)
                            else:
                                box = BoundingBox(cls_id, *map(float, splits[1:]), normalized=True)
                            boxes_list.append(box)
                    for box in boxes_list:
                        box.scale_to_size(img_width, img_height)  # TODO: Ugly

                if len(boxes_list) > 0:
                    coords_np = np.array([box.as_xyxy() for box in boxes_list], dtype=np.float32)
                    coords = torch.from_numpy(coords_np)
                    coords = torchvision.tv_tensors.BoundingBoxes(
                        coords,
                        format="XYXY",
                        canvas_size=(img_height, img_width)
                    )
                    box_labels = torch.tensor([box.cls_id for box in boxes_list], dtype=torch.long)
                else:
                    coords = torchvision.tv_tensors.BoundingBoxes(
                        torch.empty((0, 4), dtype=torch.float32),
                        format="XYXY",
                        canvas_size=(img_height, img_width)
                    )
                    box_labels = torch.empty((0,), dtype=torch.long)

                if len(line_points_list) > 0:
                    line_labels = torch.full((len(line_points_list),), self.special_classes['line_class_id'], dtype=torch.long)
                else:
                    line_labels = torch.empty((0,), dtype=torch.long)
                if len(ellipse_points_list) > 0:
                    ellipse_labels = torch.full((len(ellipse_points_list),), self.special_classes['ellipse_class_id'], dtype=torch.long)
                else:
                    ellipse_labels = torch.empty((0,), dtype=torch.long)

                if box_labels.numel() or line_labels.numel() or ellipse_labels.numel():
                    labels = torch.cat([box_labels, line_labels, ellipse_labels], dim=0)
                else:
                    labels = torch.empty((0,), dtype=torch.long)

                if len(line_points_list) > 0:
                    line_points = torch.tensor(line_points_list, dtype=torch.float32)
                    line_points[:, [0, 2]] *= img_width
                    line_points[:, [1, 3]] *= img_height
                else:
                    line_points = torch.empty((0, 4), dtype=torch.float32)

                if len(ellipse_points_list) > 0:
                    ellipses = torch.tensor(ellipse_points_list, dtype=torch.float32)
                    ellipses[:, 0] *= img_width
                    ellipses[:, 1] *= img_height
                    ellipses[:, 2] += math.log(float(img_width))
                    ellipses[:, 3] += math.log(float(img_height))
                else:
                    ellipses = torch.empty((0, 6), dtype=torch.float32)

                target = {
                    "boxes": coords,
                    "labels": labels,
                    "box_labels": box_labels,
                    "line_labels": line_labels,
                    "ellipse_labels": ellipse_labels,
                    "line_points": line_points,
                    "ellipses": ellipses,
                }
            image = torchvision.tv_tensors.Image(image) / 255.0

        if self.transforms is not None:
            if self.return_labels:
                image, target = self.transforms(image, target)
                boxes = target["boxes"]
                box_labels = target.get("box_labels", target["labels"][: len(boxes)])
                line_points = target.get("line_points", torch.empty((0, 4), dtype=torch.float32))
                ellipses = target.get("ellipses", torch.empty((0, 6), dtype=torch.float32))
                line_labels = torch.full(
                    (len(line_points),),
                    self.special_classes['line_class_id'],
                    dtype=torch.long,
                    device=line_points.device,
                )
                ellipse_labels = torch.full(
                    (len(ellipses),),
                    self.special_classes['ellipse_class_id'],
                    dtype=torch.long,
                    device=ellipses.device,
                )

                if boxes.numel() > 0:
                    valid_box_mask = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0
                else:
                    valid_box_mask = torch.zeros((0,), dtype=torch.bool, device=boxes.device)
                boxes = boxes[valid_box_mask]
                box_labels = box_labels[valid_box_mask]

                if line_points.numel() > 0:
                    h, w = image.shape[-2:]
                    x1, y1, x2, y2 = line_points.T
                    in_bounds = (
                        (x1 >= 0.0) & (x1 <= w) & (y1 >= 0.0) & (y1 <= h)
                        & (x2 >= 0.0) & (x2 <= w) & (y2 >= 0.0) & (y2 <= h)
                    )
                    lengths = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    valid_line_mask = in_bounds & (lengths > 1.0)
                    line_points = line_points[valid_line_mask]
                    line_labels = line_labels[valid_line_mask]
                else:
                    line_points = torch.empty((0, 4), dtype=torch.float32)
                    line_labels = torch.empty((0,), dtype=torch.long)

                if ellipses.numel() > 0:
                    h, w = image.shape[-2:]
                    cx, cy, log_a, log_b, cos2, sin2 = ellipses.T
                    a = torch.exp(log_a)
                    b = torch.exp(log_b)
                    valid_ellipse_mask = (
                        (cx >= 0.0) & (cx <= w) & (cy >= 0.0) & (cy <= h)
                        & torch.isfinite(log_a) & torch.isfinite(log_b)
                        & torch.isfinite(cos2) & torch.isfinite(sin2)
                        & (a > 1e-3) & (b > 1e-3)
                    )
                    ellipses = ellipses[valid_ellipse_mask]
                    ellipse_labels = ellipse_labels[valid_ellipse_mask]
                else:
                    ellipses = torch.empty((0, 6), dtype=torch.float32)
                    ellipse_labels = torch.empty((0,), dtype=torch.long)

                if box_labels.numel() or line_labels.numel() or ellipse_labels.numel():
                    target["labels"] = torch.cat([box_labels, line_labels, ellipse_labels], dim=0)
                else:
                    target["labels"] = torch.empty((0,), dtype=torch.long)
                target["boxes"] = boxes
                target["line_points"] = line_points
                target["ellipses"] = ellipses
                target["box_labels"] = box_labels
                target["line_labels"] = line_labels
                target["ellipse_labels"] = ellipse_labels

                h, w = image.shape[-2:]
                if boxes.numel() > 0:
                    target["boxes"] = boxes / torch.tensor([w, h, w, h], device=boxes.device)
                if line_points.numel() > 0:
                    target["line_points"] = line_points / torch.tensor([w, h, w, h], device=line_points.device)
                if ellipses.numel() > 0:
                    target["ellipses"][:, 0] = ellipses[:, 0] / w
                    target["ellipses"][:, 1] = ellipses[:, 1] / h
                    target["ellipses"][:, 2] -= math.log(float(w))
                    target["ellipses"][:, 3] -= math.log(float(h))
            else:
                image = self.transforms(image)
        else:
            if self.return_labels:
                h, w = image.shape[-2:]
                target["boxes"] = target["boxes"] / torch.tensor([w, h, w, h], device=target["boxes"].device)
                if target["line_points"].numel() > 0:
                    target["line_points"] = target["line_points"] / torch.tensor([w, h, w, h], device=target["line_points"].device)
                if target["ellipses"].numel() > 0:
                    target["ellipses"][:, 0] = target["ellipses"][:, 0] / w
                    target["ellipses"][:, 1] = target["ellipses"][:, 1] / h
                    target["ellipses"][:, 2] -= math.log(float(w))
                    target["ellipses"][:, 3] -= math.log(float(h))

        if self.return_labels and len(target["line_points"]) > 0:
            target["line_points"] = target["line_points"].clamp(0.0, 1.0).to(dtype=torch.float32)
        if self.return_labels and len(target["ellipses"]) > 0:
            target["ellipses"][:, :2] = target["ellipses"][:, :2].clamp(0.0, 1.0).to(dtype=torch.float32)
        return image, target if self.return_labels else image

    def get_image_path(self, idx: int) -> Path:
        """
        Get the full path to an image at the given index.

        Args:
            idx: Index of the image

        Returns:
            Full path to the image file
        """
        return Path(os.path.join(self.images_dir, self.images[idx]))

    def get_special_classes(self) -> Dict[str, int]:
        """
            Get all classes that are not standard axis-aligned bounding boxes.
            (This is used for multi-head training)
            Returns:
                A dict containing {name: class_id} pairs
        """
        return self.special_classes #{"line_class_id": self.line_class_id, "ellipse_class_id": self.ellipse_class_id}

    def get_num_classes(self) -> int:
        """
        Get the number of classes according to dataset.yaml
        """
        return self.num_classes