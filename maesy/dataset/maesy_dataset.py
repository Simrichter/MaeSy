"""Object detection dataset implementation."""

import os
from pathlib import Path

import torch
import torchvision.tv_tensors
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Dict, List, Tuple
import numpy as np
from torchvision.ops import box_convert
import yaml

from maesy.dataset.bounding_box import BoundingBox


class MaesyDataset(Dataset):
    """Dataset for object detection in COCO format."""

    def __init__(
            self,
            dataset_dir: str,
            split: str,
            dataset_type: str,
            transforms: Optional[Callable] = None,
            start_index: int = 0,
            step: int = 1,
            repeat_factor: int = 1,
            use_first_n: int = None
    ):
        """
        Initialize ObjectDetectionDataset.

        Args:
            :param dataset_dir:
            :param split: The split to be used (i.e. "train", "val", "test", etc.)
            :param dataset_type: The type of dataset, choice of ["detection", "classification", "None"]
            :param transforms: Optional transforms to apply
        """

        print("-"*60)
        print(f"Loading {split} data...")
        print("-" * 60)

        if not os.path.exists(dataset_dir):
            raise ValueError(f"Path '{dataset_dir}' does not exist.")

        if dataset_dir.endswith((".yaml", ".yml")):
            yaml_path = dataset_dir
        else:
            yaml_path = None
            yaml_candidates = [c for c in os.listdir(dataset_dir) if c.endswith((".yaml", ".yml"))]
            if len(yaml_candidates) < 1:
                print(f"Warning: No dataset.yaml file found in dataset directory {dataset_dir}\nUsing default configuration")
            elif len(yaml_candidates) > 1:
                print(f"Warning: Multiple dataset.yaml files found in dataset directory {dataset_dir}\nUsing default configuration\nFound yaml files: {yaml_candidates}\nSpecify the correct yaml directly as a path")
            else:
                yaml_path = f"{dataset_dir}/{yaml_candidates[0]}"

        # Try to read dataset configuration file
        if yaml_path is not None and os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                yaml_data = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            yaml_data = {}

        dataset_dir = yaml_data.get("path", dataset_dir)

        split_path = Path(dataset_dir) / yaml_data.get(split, split)
        assert split_path.exists(), f"Requested split {split} does not exist in dataset at {dataset_dir}"

        self.images_dir = split_path / "images"
        self.annotations_dir = split_path / "labels"
        assert self.images_dir.exists()
        assert self.annotations_dir.exists()
        self.transforms = transforms
        self.return_labels = dataset_type != "None"

        self.id_to_name = {i:v for i,v in enumerate(yaml_data.get("names", []))}
        self.name_to_id = {v:k for k,v in self.id_to_name.items()}

        self.num_classes = yaml_data.get("nc", -1)
        if self.num_classes == -1 and len(self.name_to_id) > 0:
            raise ValueError(f"Value of nc not set properly in dataset.yaml. Found {len(self.name_to_id)} classes in dataset.yaml, but no 'nc' entry")
        elif self.num_classes != len(self.name_to_id):
            raise ValueError(f"Mismatch in dataset.yaml\nnc: {self.num_classes}, but {len(self.name_to_id)} class names were found.")

        self.line_class_id = self.name_to_id.get(yaml_data.get("lines", None), -1)
        self.ellipse_class_id = self.name_to_id.get(yaml_data.get("ellipses", None), -1)

        self.images: List[Path] = [Path(img) for img in sorted(os.listdir(self.images_dir)) if
                                   img.endswith((".jpg", ".jpeg", ".png"))][start_index::step] * repeat_factor

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
                        if self.line_class_id is not None and cls_id == self.line_class_id:
                            assert len(splits) == 5, f"Invalid annotation format in {annotation_path}: '{raw_line.strip()}'. Expected 5 columns for annotation type 'line': 'class x1 y1 x2 y2'."
                            line_points_list.append([*map(float, splits[1:])])
                        elif self.ellipse_class_id is not None and cls_id == self.ellipse_class_id:
                            assert len(splits) == 6, f"Invalid annotation format in {annotation_path}: '{raw_line.strip()}'. Expected 5 columns for annotation type 'ellipse': 'class center_x center_y L_11 L_12 L_22' (cholesky decomposition)."
                            ellipse_points_list.append([*map(float, splits[1:])])
                        else:
                            assert len(
                                splits) == 5, f"Invalid annotation format in {annotation_path}: '{raw_line.strip()}'. Expected 5 columns for annotation type 'BoundingBox': 'class cx cy w h'."
                            box = BoundingBox.from_xywh(cls_id, *map(float, splits[1:]), normalized=True)
                            boxes_list.append(box)
                    for box in boxes_list:
                        box.scale_to_size(img_width, img_height)  # TODO: Ugly

                    labels_list = []
                    if len(boxes_list) > 0:
                        coords_np = np.array([box.as_cxcywh() for box in boxes_list], dtype=np.float32)
                        coords = torch.from_numpy(coords_np)

                        # Wrap as tv_tensors.BoundingBoxes with actual image size
                        coords = torchvision.tv_tensors.BoundingBoxes(
                            coords,
                            format="CXCYWH",
                            canvas_size=(img_height, img_width)  # (H, W)
                        )
                        labels_list.extend([box.cls_id for box in boxes_list])
                    else:
                        coords = torchvision.tv_tensors.BoundingBoxes(
                            torch.empty((0, 4), dtype=torch.float32),
                            format="CXCYWH",
                            canvas_size=(img_height, img_width)
                        )
                    if len(line_points_list) > 0:
                        labels_list.extend([self.line_class_id] * len(line_points_list))
                    if len(ellipse_points_list) > 0:
                        labels_list.extend([self.ellipse_class_id] * len(ellipse_points_list))
                    labels = torch.tensor(labels_list, dtype=torch.long)

                    # Basically a "combined" else to the three above
                    if len(boxes_list) <= 0 and len(line_points_list) <= 0:
                        # coords = torchvision.tv_tensors.BoundingBoxes(
                        #     torch.empty((0, 4), dtype=torch.float32),
                        #     format="CXCYWH",
                        #     canvas_size=(img_height, img_width)
                        # )
                        labels = torch.empty((0,), dtype=torch.long)

                    if len(line_points_list) > 0:
                        line_points = torch.tensor(line_points_list, dtype=torch.float32)
                    else:
                        line_points = torch.empty((0, 4), dtype=torch.float32)

                    if len(ellipse_points_list) > 0:
                        ellipses = torch.tensor(ellipse_points_list, dtype=torch.float32)
                    else:
                        ellipses = torch.empty((0, 5), dtype=torch.float32)

                    target = {"boxes": coords, "labels": labels, "line_points": line_points, "ellipses": ellipses}
            image = torchvision.tv_tensors.Image(image) / 255.0

            if self.transforms is not None:
                if self.return_labels:
                    image, target = self.transforms(image, target)
                    # Filter out invalid boxes (out of bounds or zero area)
                    boxes = target["boxes"]
                    labels = target["labels"]

                    # Convert to xyxy to check validity
                    boxes_xyxy = box_convert(boxes, "cxcywh", "xyxy")

                    # Keep boxes that have area > 0
                    valid_mask = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) > 0

                    target["boxes"] = boxes[valid_mask]
                    if target["line_points"].shape[0] > 0:
                        valid_mask = torch.cat((valid_mask, torch.ones((target["line_points"].shape[0],), dtype=torch.bool,
                                                                       device=valid_mask.device)))  # Keep all line targets in labels
                    if target["ellipses"].shape[0] > 0:
                        valid_mask = torch.cat((valid_mask, torch.ones((target["ellipses"].shape[0],), dtype=torch.bool, device=valid_mask.device))) # Keep all ellipse targets in labels
                    target["labels"] = labels[valid_mask]

                    # Normalize boxes back to [0,1]
                    h, w = image.shape[-2:]
                    if len(target["boxes"]) > 0:
                        target["boxes"] = target["boxes"] / torch.tensor([w, h, w, h], device=target["boxes"].device)
                else:
                    image = self.transforms(image)
            else:
                # Default: convert image to tensor
                # image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                if self.return_labels:
                    # Normalize boxes back to [0,1] after transforms
                    h, w = image.shape[-2:]
                    target["boxes"] = target["boxes"] / torch.tensor([w, h, w, h], device=target["boxes"].device)

            if self.return_labels and len(target["line_points"]) > 0:
                target["line_points"] = target["line_points"].clamp(0.0, 1.0).to(dtype=torch.float32)
            # if self.return_labels and len(target["ellipses"]) > 0:
            #     target["ellipses"][:, :2] = target["ellipses"][:, :2].clamp(0.0, 1.0).to(dtype=torch.float32)
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
                A dict containing name: class_id pairs
        """
        return {"line_class_id": self.line_class_id, "ellipse_class_id": self.ellipse_class_id}

    def get_num_classes(self) -> int:
        """
        Get the number of classes according to dataset.yaml
        """
        return self.num_classes