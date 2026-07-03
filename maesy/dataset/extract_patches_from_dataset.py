"""
 Provides utility to extract patches of arbitrary classes from MaesyDatasets (supports transforms)
"""
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from maesy.dataset import MaesyDataset
from maesy.training.utils import collate_detection_fn

import numpy as np

from torch.utils.data import DataLoader

class Patch:
    """
        A small representation of a patch that automatically loads and crops itself
    """
    def __init__(self, position_in_image: Tuple[float, float, float, float], original_image_path):
        self.position_in_image = position_in_image
        self.original_image_path = original_image_path

    def save_patch_to(self, save_path: Path, size=(24, 48)):
        """
        Save the patch to the given path.
        Utilizes lazy-loading of image data only when it is requested
        """
        with (Image.open(self.original_image_path) as img):
            w, h = img.size
            x1, x2 = self.position_in_image[0]*w, self.position_in_image[2]*w
            y1, y2 = self.position_in_image[1]*h, self.position_in_image[3]*h
            img = img.crop((x1, y1, x2, y2))
            img = img.resize(size)
            img.save(save_path)

def _save_patches(patches: List[Patch], save_path: str, cls: int):
    for i, p in enumerate(patches):
        p.save_patch_to(Path(save_path) / f"patch_{Path(p.original_image_path).name.split('.')[0]}_{cls}.png")

def _overlap_with_existing(box, boxes_in_image, threshold=0.0):
    """
        Check if the box overlaps with any existing boxes in the image or false positive boxes
    """
    for existing_box in boxes_in_image:
        x1, y1, x2, y2 = box
        ex1, ey1, ex2, ey2 = existing_box
        # Calculate intersection area
        inter_x1 = max(x1, ex1)
        inter_y1 = max(y1, ey1)
        inter_x2 = min(x2, ex2)
        inter_y2 = min(y2, ey2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        if threshold == 0.0: # full IoU not needed here
            if inter_area>0.0:
                return True
            else:
                continue

        # Calculate union area
        box_area = (x2 - x1) * (y2 - y1)
        existing_box_area = (ex2 - ex1) * (ey2 - ey1)
        union_area = box_area + existing_box_area - inter_area

        # Calculate IoU
        iou = inter_area / union_area if union_area > 0 else 0

        if iou > threshold:
            return True
    return False

def _generate_patches_from_dataset(dataset: MaesyDataset, desired_class_id, output_fp: bool, margin: float) -> Tuple[List[Patch], List[Patch]]:
    """
        Collect all patches of the desired class that are present in the Dataset

        Args:
            :param dataset: The source MaesyDataset
            :param desired_class_id: The id of the objects to extract
            :param output_fp: Whether to generate random patches that do not overlap correct patches
            :param margin: A percent-value to be added as symmetric margin around the borders. Relative to width and height of object
    """
    batch_size = 1
    loader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_detection_fn)

    patch_list = []
    patches_per_image = 0
    fp_list = []
    for idx, (imgs, targets) in enumerate(tqdm(loader)):
        path = dataset.get_image_path(idx)
        for img, objects in zip(imgs, targets):
            patches_per_image = 0
            boxes_in_image = []
            for label, box in zip(objects["labels"], objects["boxes"]):
                if label.item() == desired_class_id:
                    x1, y1, x2, y2 = box[0].item(), box[1].item(), box[2].item(), box[3].item()
                    w, h = x2-x1, y2-y1
                    x1, y1, x2, y2 = x1 - margin*w, y1 - margin*h, x2 + margin*w, y2 + margin*h
                    borders = (x1, y1, x2, y2)
                    patch_list.append(Patch(borders, path))
                    patches_per_image += 1
                    if output_fp:
                        boxes_in_image.append(borders)
            if output_fp:
                num_fp_boxes = 0
                retries = 0
                fp_boxes = []
                while num_fp_boxes < patches_per_image and retries < 100:
                    box = np.random.rand(4)
                    box[2:].sort() # Optional (maybe introduce a ratio factor)
                    box = tuple((*box[:2].clip(0.0, 1.0), *(box[:2] + box[2:]/2).clip(0.0, 1.0)))
                    if not _overlap_with_existing(box, boxes_in_image, threshold=0.0) and not _overlap_with_existing(box, fp_boxes, threshold=0.25):
                        fp_boxes.append(box)
                        fp_list.append(Patch(box, path))
                        num_fp_boxes += 1
                    else:
                        retries += 1

    return patch_list, fp_list

def extract_patches(dataset_path: str, split: List[str], patch_path: str, desired_class_id: int, output_fp: bool, margin: float):
    """
    Extract patches of selected class from MaesyDataset
    """
    for s in split:
        try:
            dataset = MaesyDataset(dataset_path, s, annotation_type="detection")
        except Exception as e:
            print(f"Failed to load dataset for split '{s}'")
            continue
        if len(dataset) == 0:
            print(f"No content, skipping {s}...")
            continue
        print("Extracting...")
        op = patch_path
        if op == "":
            op = f"{dataset_path}/extracted_patches/{s}/images"
            Path(op).mkdir(parents=True, exist_ok=True)
        robot_patches, fp_patches = _generate_patches_from_dataset(dataset, desired_class_id, output_fp=output_fp, margin=margin)
        _save_patches(robot_patches, op, 1)
        if output_fp:
            _save_patches(fp_patches, op, 0)
