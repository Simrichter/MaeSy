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


def _load_yaml_data(dataset_dir: str, annotation_type: str) -> Tuple[dict, str]:
    """
    auto-search yaml, auto-infer annotation type if not specified (currently only 'image_folder')
    """
    if annotation_type == "image_folder":
        return {}, "image_folder"
    if dataset_dir.endswith((".yaml", ".yml")):
        yaml_path = dataset_dir
    else:
        yaml_candidates = [c for c in os.listdir(dataset_dir) if c.endswith((".yaml", ".yml"))]
        if len(yaml_candidates) < 1:
            if annotation_type in ["auto", "image_folder"]:
                print(f"Annotation type image_folder detected")
                return {}, "image_folder" # Return minimal yaml data for image_folder structure
            else:
                raise ValueError(f"Error: No dataset.yaml file found in dataset directory {dataset_dir}, but annotation type {annotation_type} was specified")
        elif len(yaml_candidates) > 1:
            raise ValueError(
                f"Multiple dataset.yaml files found in dataset directory {dataset_dir}\n"
                f"Found yaml files: {yaml_candidates}\n"
                f"Specify the correct yaml file directly as the dataset path")
        else:
            yaml_path = f"{dataset_dir}/{yaml_candidates[0]}"

    # Try to read dataset configuration file
    if yaml_path is not None and os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            return yaml.load(f, Loader=yaml.SafeLoader), annotation_type
    else:
        raise FileNotFoundError(f"Failed to read file {yaml_path}!")

def _infer_annotation_type(dataset_dir, yaml_data: dict) -> str:
    """
    Check for entry in yaml. If not present, interpret existing labels folder as 'detection', otherwise 'None'
    """
    if "annotation_type" in yaml_data:
        annotation_type = yaml_data["annotation_type"]
        print(f"Inferred annotation type '{annotation_type}' from dataset.yaml")
        return annotation_type
    else:
        if (Path(dataset_dir) / "labels").exists():
            print("Found 'labels' folder in dataset directory. Assuming annotation type 'detection'.")
            return "detection"
        else:
            print("No 'labels' folder found in dataset directory. Assuming annotation type 'None'.")
            return "None"

class MaesyDataset(Dataset):
    """Dataset for object detection in COCO format."""

    def __init__(
            self,
            dataset_dir: str,
            split: str = "None",
            annotation_type: str = "None",
            transforms: Optional[Callable] = None,
            start_index: int = 0,
            step: int = 1,
            enable_lines: bool = False,
            enable_ellipses: bool = False,
            use_first_n: Optional[int] = None
    ):
        """
        Initialize ObjectDetectionDataset.

        Args:
            :param dataset_dir: Path to the MaesyDataset root directory
            :param split: The split to be used (i.e. "train", "val", "test", etc.)
            :param annotation_type: The type of dataset, choice of ["detection", "None", "image_folder", "auto"] ("image_folder" assumes no labels) # "classification" might come in future
            :param transforms: Optional transforms to apply
            :param start_index: The index from which to start
            :param step: The step size for sampling images (e.g., step=2 will take every other image)
            # :param repeat_factor: The factor by which to repeat the dataset (e.g., repeat_factor=2 will repeat the dataset twice, effectively doubling its size)
            :param enable_lines: Whether to include line annotations (if line_class_id is defined in dataset.yaml)
            :param enable_ellipses: Whether to include ellipse annotations (if ellipse_class_id is defined in dataset.yaml)
            :param use_first_n: If not None, only use the first n samples from the dataset (after applying start_index, step, repeat_factor)
        """

        if not os.path.exists(dataset_dir):
            raise ValueError(f"Path '{dataset_dir}' does not exist.")

        self.transforms = transforms

        yaml_data, annotation_type = _load_yaml_data(dataset_dir, annotation_type)
        if annotation_type == "auto":
            annotation_type = _infer_annotation_type(dataset_dir, yaml_data)

        if annotation_type == "image_folder":
            self.repeat_factor = 1 # image folders currently don't support repetition (because factor is set in dataset.yaml)
            self._load_image_folder(dataset_dir)
        else:
            if split == "train":  # Only repeat if in train mode. Validation and testing should remain unrepeated
                self.repeat_factor = yaml_data.get("repeat_factor", 1)  # Default value is 1, but could be increased for overfitting tests
            else:
                self.repeat_factor = 1
            self._load_dataset(yaml_data, split, annotation_type, enable_lines, enable_ellipses)

        self.images = self.images[start_index::step] * self.repeat_factor
        if self.return_labels and self.annotations:
            self.annotations = self.annotations[start_index::step] * self.repeat_factor

        if use_first_n is not None:
            self.images = self.images[:use_first_n]
            self.annotations = self.annotations[:use_first_n]

    def _load_dataset(self, yaml_data: dict, split: str, annotation_type: str, enable_lines: bool, enable_ellipses: bool):
        assert split != "None", "Failed: Split must be specified for dataset types 'detection', 'classification' or 'None'!\nOnly type 'image_folder' does not require a split!"

        dataset_dir = yaml_data.get("path")
        assert dataset_dir is not None, "Failed: Could not find information about data path in yaml file!"

        self.box_format = str(yaml_data.get("box_format", "")).lower()
        if self.box_format not in {"xyxy", "cxcywh"} and annotation_type == "detection":
            raise ValueError(f"Unsupported box_format '{self.box_format}' in dataset.yaml. Expected 'xyxy' or 'cxcywh'.")

        split_path = Path(dataset_dir) / yaml_data.get(split, split)
        assert split_path.exists(), f"Requested split '{split}' does not exist in dataset at {dataset_dir}"

        self.return_labels = annotation_type not in ["None", "image_folder"]

        self.images_dir = split_path / "images"
        assert self.images_dir.exists()

        self.images: List[Path] = [Path(img) for img in sorted(os.listdir(self.images_dir)) if img.endswith((".jpg", ".jpeg", ".png"))]

        if self.return_labels:
            self.annotations_dir = split_path / "labels"
            assert self.annotations_dir.exists()

            self.id_to_name = {i: v for i, v in enumerate(yaml_data.get("names", []))}
            self.name_to_id = {v: k for k, v in self.id_to_name.items()}
            self.num_classes = yaml_data.get("nc", -1)
            if self.num_classes == -1 and len(self.name_to_id) > 0:
                raise ValueError(f"Value of nc not set properly in dataset.yaml. Found {len(self.name_to_id)} classes in dataset.yaml, but no 'nc' entry")
            elif self.num_classes != -1 and self.num_classes != len(self.name_to_id):
                raise ValueError(f"Mismatch in dataset.yaml\nnc: {self.num_classes}, but {len(self.name_to_id)} class names were found.")

            self.special_classes = {'line_class_id': self.name_to_id.get(yaml_data.get("lines"), -1),
                                    'ellipse_class_id': self.name_to_id.get(yaml_data.get("ellipses"), -1)}
            self.enable_lines = enable_lines and self.special_classes["line_class_id"] != -1
            self.enable_ellipses = enable_ellipses and self.special_classes["ellipse_class_id"] != -1

            # Correct num_classes count if lines/ellipses are disabled, but would have been present in the dataset otherwise
            if not self.enable_lines and self.special_classes["line_class_id"] != -1:
                self.num_classes -= 1
            if not self.enable_ellipses and self.special_classes["ellipse_class_id"] != -1:
                self.num_classes -= 1

            self.annotations = []
            for img in self.images:
                annotation_path = Path(img).with_suffix(".txt")
                if not (self.annotations_dir / annotation_path).exists():
                    raise FileNotFoundError(f"Annotation file {annotation_path} not found for image {img}")
                self.annotations.append(annotation_path)
        print(f"Loaded {len(self.images)} {split} images")
        print("-" * 30)

    def _load_image_folder(self, dataset_dir):
        assert os.path.exists(dataset_dir), f"Failed: Image folder {dataset_dir} does not exist!"
        self.return_labels = False
        self.images_dir = dataset_dir
        self.images: List[Path] = [Path(img) for img in sorted(os.listdir(dataset_dir)) if img.endswith((".jpg", ".jpeg", ".png"))]

        print(f"Loaded image folder from path {dataset_dir}...")
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
                    # Keep all annotation coordinates normalized in [0,1].
                    # BoundingBox objects were created with normalized=True, so do NOT
                    # scale to pixel coordinates here (transforms may resize later).
                    # Convert to plain tensors (Nx4) in XYXY normalized format.
                    if len(boxes_list) > 0:
                        coords_np = np.array([box.as_xyxy() for box in boxes_list], dtype=np.float32)
                        coords = torch.from_numpy(coords_np)
                        box_labels = torch.tensor([box.cls_id for box in boxes_list], dtype=torch.long)
                    else:
                        coords = torch.empty((0, 4), dtype=torch.float32)
                        box_labels = torch.empty((0,), dtype=torch.long)

                # Create label tensors for lines and ellipses (labels are class ids)
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

                # Line points are expected normalized [x1 y1 x2 y2] in dataset. Keep them normalized.
                if len(line_points_list) > 0:
                    line_points = torch.tensor(line_points_list, dtype=torch.float32)
                else:
                    line_points = torch.empty((0, 4), dtype=torch.float32)

                # Ellipses are expected normalized in dataset: [cx cy log_a log_b cos2 sin2]
                # Keep as normalized representation. Validate finite/logical values later.
                if len(ellipse_points_list) > 0:
                    ellipses = torch.tensor(ellipse_points_list, dtype=torch.float32)
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

            # Apply transforms (if any). We intentionally keep all geometric
            # annotations normalized in [0,1] so transforms should accept that.
            if self.transforms is not None:
                if self.return_labels:
                    image, target = self.transforms(image, target)
                    boxes = target["boxes"]
                    box_labels = target.get("box_labels", target["labels"][: len(boxes)])
                    line_points = target.get("line_points", torch.empty((0, 4), dtype=torch.float32))
                    ellipses = target.get("ellipses", torch.empty((0, 6), dtype=torch.float32))
                    line_labels = torch.full((len(line_points),), self.special_classes['line_class_id'], dtype=torch.long, device=line_points.device)
                    ellipse_labels = torch.full((len(ellipses),), self.special_classes['ellipse_class_id'], dtype=torch.long, device=ellipses.device)

                    # sanitize boxes (assume normalized XYXY)
                    if boxes.numel() > 0:
                        valid_box_mask = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0
                    else:
                        valid_box_mask = torch.zeros((0,), dtype=torch.bool, device=boxes.device)
                    boxes = boxes[valid_box_mask]
                    box_labels = box_labels[valid_box_mask]

                    # Clip and sanitize line segments using normalized coords.
                    def liang_barsky_clip(x0, y0, x1, y1, eps=1e-9):
                        dx = x1 - x0
                        dy = y1 - y0
                        p = [-dx, dx, -dy, dy]
                        q = [x0 - 0.0, 1.0 - x0, y0 - 0.0, 1.0 - y0]
                        u1 = 0.0
                        u2 = 1.0
                        for pi, qi in zip(p, q):
                            if abs(pi) < 1e-12:
                                if qi < 0.0:
                                    return None
                                else:
                                    continue
                            t = qi / pi
                            if pi < 0:
                                if t > u2:
                                    return None
                                if t > u1:
                                    u1 = t
                            else:
                                if t < u1:
                                    return None
                                if t < u2:
                                    u2 = t
                        if u2 < u1:
                            return None
                        nx0 = x0 + u1 * dx
                        ny0 = y0 + u1 * dy
                        nx1 = x0 + u2 * dx
                        ny1 = y0 + u2 * dy
                        nx0 = 0.0 if abs(nx0) < eps else (1.0 if abs(nx0 - 1.0) < eps else nx0)
                        ny0 = 0.0 if abs(ny0) < eps else (1.0 if abs(ny0 - 1.0) < eps else ny0)
                        nx1 = 0.0 if abs(nx1) < eps else (1.0 if abs(nx1 - 1.0) < eps else nx1)
                        ny1 = 0.0 if abs(ny1) < eps else (1.0 if abs(ny1 - 1.0) < eps else ny1)
                        if (nx1 - nx0) ** 2 + (ny1 - ny0) ** 2 < (eps ** 2):
                            return None
                        return [nx0, ny0, nx1, ny1]

                    if line_points.numel() > 0:
                        clipped_lines = []
                        clipped_labels = []
                        for lp, lab in zip(line_points.tolist(), line_labels.tolist()):
                            x1, y1, x2, y2 = lp
                            eps = 1e-9
                            x1 = 0.0 if abs(x1) < eps else x1
                            y1 = 0.0 if abs(y1) < eps else y1
                            x2 = 0.0 if abs(x2) < eps else x2
                            y2 = 0.0 if abs(y2) < eps else y2
                            res = liang_barsky_clip(x1, y1, x2, y2, eps=eps)
                            if res is not None:
                                clipped_lines.append(res)
                                clipped_labels.append(lab)
                        if len(clipped_lines) > 0:
                            line_points = torch.tensor(clipped_lines, dtype=torch.float32)
                            line_labels = torch.tensor(clipped_labels, dtype=torch.long)
                        else:
                            line_points = torch.empty((0, 4), dtype=torch.float32)
                            line_labels = torch.empty((0,), dtype=torch.long)

                    else:
                        line_points = torch.empty((0, 4), dtype=torch.float32)
                        line_labels = torch.empty((0,), dtype=torch.long)

                    # Validate ellipses.
                    if ellipses.numel() > 0:
                        cx = ellipses[:, 0]
                        cy = ellipses[:, 1]
                        log_a = ellipses[:, 2]
                        log_b = ellipses[:, 3]
                        cos2 = ellipses[:, 4]
                        sin2 = ellipses[:, 5]
                        eps = 1e-9
                        cx = torch.where(torch.abs(cx) < eps, torch.zeros_like(cx), cx)
                        cy = torch.where(torch.abs(cy) < eps, torch.zeros_like(cy), cy)
                        centers_in = (cx >= 0.0) & (cx <= 1.0) & (cy >= 0.0) & (cy <= 1.0)
                        finite_mask = torch.isfinite(log_a) & torch.isfinite(log_b) & torch.isfinite(cos2) & torch.isfinite(sin2)
                        a = torch.exp(log_a)
                        b = torch.exp(log_b)
                        size_mask = (a > 1e-6) & (b > 1e-6)
                        valid_ellipse_mask = centers_in & finite_mask & size_mask
                        if valid_ellipse_mask.any():
                            ellipses = ellipses[valid_ellipse_mask]
                            ellipse_labels = ellipse_labels[valid_ellipse_mask]
                        else:
                            ellipses = torch.empty((0, 6), dtype=torch.float32)
                            ellipse_labels = torch.empty((0,), dtype=torch.long)
                    else:
                        ellipses = torch.empty((0, 6), dtype=torch.float32)
                        ellipse_labels = torch.empty((0,), dtype=torch.long)

                    # Write back to target
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
                else:
                    # transforms provided but no labels requested
                    image = self.transforms(image)
            else:
                # No transforms provided. Annotations are already normalized; just clamp small numerical noise.
                if self.return_labels:
                    if target.get("line_points", torch.empty((0, 4))).numel() > 0:
                        target["line_points"] = target["line_points"].clamp(0.0, 1.0).to(dtype=torch.float32)
                    if target.get("ellipses", torch.empty((0, 6))).numel() > 0:
                        target["ellipses"][:, :2] = target["ellipses"][:, :2].clamp(0.0, 1.0).to(dtype=torch.float32)
                # image remains as-is

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
        if self.return_labels:
            return self.special_classes #{"line_class_id": self.line_class_id, "ellipse_class_id": self.ellipse_class_id}
        else:
            print("Warning: get_special_classes call to a dataset without labels!")
            return {}

    def get_num_classes(self) -> int:
        """
        Get the number of classes according to dataset.yaml
        Returns:
            Number of classes
        """
        if self.return_labels:
            return self.num_classes
        else:
            print("Warning: get_num_classes call to a dataset without labels!")
            return 0