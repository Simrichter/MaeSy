import ast
import json
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from skimage.measure import EllipseModel
from sklearn.decomposition import PCA
from tqdm import tqdm

def ellipse_to_cholesky(cx: float, cy: float, a: float, b: float, theta: float):
    """
    Convert ellipse parameters (center, axes, angle) to a Cholesky-like representation suitable for regression.
    The output format is [xc, yc, l11, l21, l22]

    Args:
        :param cx: x-coordinate of ellipse center
        :param cy: y-coordinate of ellipse center
        :param a: Length of one diagonal
        :param b: Length of other diagonal
        :param theta: Rotation angle of main diagonal
    """
    # optional safety (recommended)
    a = np.clip(a, 1e-6, None)
    b = np.clip(b, 1e-6, None)

    c = np.cos(theta)
    s = np.sin(theta)

    inv_a2 = 1.0 / (a * a)
    inv_b2 = 1.0 / (b * b)

    A11 = c*c * inv_a2 + s*s * inv_b2
    A12 = c*s * (inv_a2 - inv_b2)
    A22 = s*s * inv_a2 + c*c * inv_b2

    l11 = np.sqrt(A11)
    l21 = A12 / l11

    # numerical stability
    tmp = A22 - l21*l21
    l22 = np.sqrt(np.clip(tmp, 1e-8, None))

    return np.stack([cx, cy, l11, l21, l22], axis=-1)

def robert_to_devils_yolo(dataset_dir: str | Path, class_id_blacklist: Optional[list[int]]=None, merge_ids: Optional[dict[int, int]]=None, permute_ids: Optional[list[int]]=None):
    """
        Convert Image annotations from Robert's Unity simulator outputs to devils_yolo format.
        Returns output path, number of classes, class_names and special_classes dict (for create dataset)

        Args:
            :param dataset_dir: Path to the root data folder
            :param class_id_blacklist: List of class ids to ignore during conversion
            :param merge_ids: A dict of {x: int, y: int} pairs, where x is the class ID to be merged into class ID y
            :param permute_ids: A list of indices used to permute the class IDs. The index of the list represents the old class ID, the value at that index represents the new class ID. This is applied after merging and blacklisting operations.

        Returns:
            path: Path to images and labels
            nc: Number of classes
            class_names: List of class names
            special_classes: Dict of special classes (lines, ellipses)
    """

    print("="*60)
    print(f"Converting Robert's Unity Simulator annotations to DevilsYolo format")
    print(f"Dataset path: {dataset_dir}")
    print("=" * 60)

    name_to_id = {
        "Trionda Ball 2026(Clone)": 0,
        "K1(Clone)": 1,
        "PenaltyCross": 2,
        "Line": 3,
        "CenterCircle": 4,
    }

    if merge_ids is None:
        merge_ids = {}

    if class_id_blacklist is None:
        class_id_blacklist = {}

    if permute_ids is None or len(permute_ids) != len(set(name_to_id.values())):
        permute_ids = range(len(name_to_id.values()))

    # apply merging and blacklisting operations
    id_to_name = {v: k for k, v in name_to_id.items()}
    for x, y in merge_ids.items():
        name_to_id[id_to_name[x]] = y
    for class_id in class_id_blacklist:
        del name_to_id[id_to_name[class_id]]
    # rebuild indices
    unique_ids = list(name_to_id.fromkeys(name_to_id.values()))
    unique_ids[:] = [unique_ids[i] for i in permute_ids]
    translate_ids = {v: k for k, v in enumerate(unique_ids)}
    name_to_id = {k: translate_ids[v] for k, v in name_to_id.items()}
    # build special classes dict
    special_classes = {}
    if "Line" in name_to_id.keys():
        special_classes["lines"] = "Line"
    if "CenterCircle" in id_to_name.values():
        special_classes["ellipses"] = "CenterCircle"

    dataset_dir: Path = Path(dataset_dir)
    splits = [f for f in os.listdir(dataset_dir) if f in ["train", "val", "test"]]
    print(f"Found {len(splits)} splits: {splits}")
    for split in splits:
        print(f"Handling {split}...")
        path = dataset_dir / split / "labels"
        if not os.path.exists(path):
            raise ValueError(f"Could not find '{split}/labels' Folder in dataset root!!")
        out_path: Path = dataset_dir / split / "images"
        out_path.mkdir(exist_ok=True)

        txts = [t for t in os.listdir(path) if t.endswith(".txt")]
        ellipse_model = EllipseModel()
        for label_file in tqdm(txts):
            with open(os.path.join(path, label_file), 'r') as f:
                lines = f.readlines()
            new_lines = []
            current_cat = None
            for line in lines:
                cat = line.strip().removesuffix(":")
                if cat in ["BoundingBoxes", "GoalPosts", "CenterCircle", "PenaltyPoints", "Lines"]:
                    current_cat = cat
                    continue
                match current_cat:
                    case "BoundingBoxes":
                        class_name, coords = line.strip().split(":")
                        class_id = name_to_id[class_name]
                        class_id = merge_ids.get(class_id, class_id)
                        if class_id is not None and class_id in class_id_blacklist:
                            continue
                        parts = coords.strip().split()
                        if len(parts) != 4:
                            print(f"Skipping BoundingBox line in file {label_file} due to incorrect format: {line}")
                            continue
                        cx, cy, w, h = parts
                        cx, cy, w, h = float(cx), float(cy), float(w), float(h)  # Explicit cast to ensure datatype compatibility
                        x1 = cx - 0.5 * w
                        y1 = cy - 0.5 * h
                        x2 = cx + 0.5 * w
                        y2 = cy + 0.5 * h
                        new_lines.append(f"{class_id} {x1} {y1} {x2} {y2}\n")
                    case "Lines":
                        class_id = name_to_id['Line']
                        class_id = merge_ids.get(class_id, class_id)
                        if class_id is not None and class_id in class_id_blacklist:
                            continue
                        line_parts = line.strip().split(", ")
                        if len(line_parts) == 4: # Subtype: Line
                            x1, y1, x2, y2 = [float(lp.lstrip("(").rstrip("),")) for lp in line_parts]
                            x1, x2 = x1/544, x2/544
                            y1, y2 = 1-y1/448, 1-y2/448 # TODO: Quickfix for horizontally flipped lines????
                            new_lines.append(f"{class_id} {x1} {y1} {x2} {y2}\n") # Class ID 1 for lines, can be changed if needed
                        else: # Subtype: CornerArc
                            ... # TODO
                    case "PenaltyPoints":
                        class_id = name_to_id['PenaltyCross']
                        class_id = merge_ids.get(class_id, class_id)
                        if class_id is not None and class_id in class_id_blacklist:
                            continue
                        parts = line.strip().split()
                        if len(parts) != 4:
                            print(f"Skipping PenaltyPoint line in file {label_file} due to incorrect format: {line}")
                            continue
                        cx, cy, w, h = parts
                        cx, cy, w, h = float(cx), float(cy), float(w), float(h)  # Explicit cast to ensure datatype compatibility
                        x1 = cx - 0.5 * w
                        y1 = cy - 0.5 * h
                        x2 = cx + 0.5 * w
                        y2 = cy + 0.5 * h
                        new_lines.append(f"{class_id} {x1} {y1} {x2} {y2}\n")
                    case "CenterCircle":
                        class_id = name_to_id['CenterCircle']
                        class_id = merge_ids.get(class_id, class_id)
                        if class_id is not None and class_id in class_id_blacklist:
                            continue
                        points = np.array(ast.literal_eval(f"[{line}]"))
                        points[:,::2] = points[:,::2]/544
                        points[:,1::2] = 1-points[:,1::2]/448
                        if ellipse_model.estimate(points):
                            cx, cy, a, b, theta = ellipse_model.params
                            new_lines.append(f"{class_id} {cx} {cy} {np.log(a)} {np.log(b)} {np.cos(2*theta)} {np.sin(2*theta)}\n")
                        else:
                            print(f"Failed to estimate ellipse from line '{line}' in file {label_file}")
                    case _:
                        # ignore other categories for now
                        ...

            with open(os.path.join(out_path, label_file), 'w') as f:
                f.writelines(new_lines)

    return out_path, len(name_to_id.values()), list(name_to_id.keys()), special_classes


def datumaro_to_devils_yolo(datumaro_dir: str, class_id_blacklist: Optional[list[int]]=None, merge_ids: Optional[dict[int, int]]=None, permute_ids: Optional[list[int]]= None) -> Tuple[str, int, list, dict]:
        """
        Convert Image annotations exported from cvat in datumaro JSON format into DevilsYolo format.
        Lines are represented by their two endpoint coordinates in xyxy format.
        All coordinates are normalized
        Returns output path, number of classes, class_names and special_classes dict (for create dataset)

        Args:
            :param datumaro_dir: Path to the datumaro dataset root folder (the one that contains the annotations folder and the images folder)
            :param class_id_blacklist: List of class ids to ignore during conversion
            :param merge_ids: A dict of {x: int, y: int} pairs, where x is the class ID to be merged into class ID y (only class ID y is kept afterward)
            :param permute_ids: A list of indices used to permute the class IDs. The index of the list represents the old class ID, the value at that index represents the new class ID. This is applied after merging and blacklisting operations.

        Returns:
            path: Path to images and labels
            nc: Number of classes
            class_names: List of class names
            special_classes: Dict of special classes (lines, ellipses)
        """

        print("="*60)
        print(f"Converting datumaro to DevilsYolo format")
        print(f"Dataset root path: {datumaro_dir}")
        print("=" * 60)

        def _fix_multipoint_line(datadict):
            for (img_i, img_dat) in enumerate(datadict['items']):
                anns_to_be_added = []
                for ann_i, annot in enumerate(img_dat['annotations']):
                    if annot['type'] == "polyline":
                        points = annot['points']
                        if len(points) != 4:
                            print(f"Fixing Polyline for img: {img_dat['id']} with {len(points) / 2 - 1} lines")
                            for i in range(0, len(points) - 3, 2):
                                p1 = (points[i], points[i + 1])
                                p2 = (points[i + 2], points[i + 3])
                                new_ann = annot.copy()
                                new_ann["points"] = [p1[0], p1[1], p2[0], p2[1]]
                                if not (p1[0] == p2[0] and p1[1] == p2[1]):  # Check for zero-length vector
                                    anns_to_be_added.append(new_ann)
                                else:
                                    print(f"Skipping double point for annotation {ann_i} in file {img_dat['id']} ({img_i}) due to zero-length vector: {p1} and {p2}")

                            annot['points'] = anns_to_be_added[0]['points']
                            anns_to_be_added = anns_to_be_added[1:]
                img_dat['annotations'].extend(anns_to_be_added)

        with open(f"{datumaro_dir}/annotations/default.json", 'r') as f:
            data = json.load(f)
        print("JSON loaded successfully")
        print("Checking for multi-point line annotations and fixing them if necessary...")
        _fix_multipoint_line(data)

        img_dir = f"{datumaro_dir}/images/default"
        ellipse_model = EllipseModel()

        id_to_name = {i: label["name"] for i, label in enumerate(data["categories"]["label"]["labels"])}

        if merge_ids is None:
            merge_ids = {}
        if class_id_blacklist is None:
            class_id_blacklist = []

        # update merge and blacklist information into id_to_name to return the correct nc, class_names and special_classes
        for k, v in merge_ids.items():
            print(f"Merging class {id_to_name[k]} into class {id_to_name[v]}")
            id_to_name[k] = id_to_name[v]
        for class_id in class_id_blacklist:
            print(f"Removing {class_id} ({id_to_name[class_id]}) from id_to_name due to blacklist")
            del id_to_name[class_id]
        # Rebuilding keys to ensure continuous class IDs and no duplicate names
        unique_names = list(id_to_name.fromkeys(id_to_name.values()))
        if permute_ids is None or len(permute_ids) == 0:
            permute_ids = range(len(unique_names))
        else:
            assert len(permute_ids) == len(unique_names), f"Error: number of permute_ids ({len(permute_ids)}) does not match number of unique class names ({len(unique_names)}) after merging and blacklisting operations."
        unique_names[:] = [unique_names[i] for i in permute_ids or range(len(unique_names))]
        new_id_to_name = {k: v for k, v in enumerate(unique_names)} # Only used to generate information for building the dataset.yaml file
        name_to_new_id = {v: k for k, v in new_id_to_name.items()}
        translate_ids = {k: name_to_new_id[v] for k, v in id_to_name.items()}  # Usage: translate_ids[old_id] = new_id
        id_to_name = new_id_to_name
        # Building special_classes if still present in dict
        special_classes = {}
        if "Lines" in id_to_name.values():
            special_classes["lines"] = "Lines"
        if "CenterCircle" in id_to_name.values():
            special_classes["ellipses"] = "CenterCircle"

        print("Writing .txt annotations...")
        for img_ind, img_data in enumerate(tqdm(data['items'])):
            with (open(os.path.join(img_dir, img_data['id'] + ".txt"), 'w') as f):
                normalize = img_data["image"]["size"]
                normalize.reverse() # datumaro saves height, width
                for ann_ind, ann in enumerate(img_data['annotations']):
                    class_id = ann['label_id']
                    class_id = merge_ids.get(class_id, class_id)
                    if class_id in class_id_blacklist:
                        continue
                    if ann['type'] == "bbox": # BoundingBoxes
                        x1 = max(ann['bbox'][0] / normalize[0], 0.0)
                        y1 = max(ann['bbox'][1] / normalize[1], 0.0)
                        x2 = min((ann['bbox'][0] + ann['bbox'][2]) / normalize[0], 1.0)
                        y2 = min((ann['bbox'][1] + ann['bbox'][3]) / normalize[1], 1.0)
                        f.write(f"{translate_ids[class_id]} {x1} {y1} {x2} {y2}\n")
                    elif ann['type'] == "polyline": # FieldLines
                        p = ann['points']
                        if len(p) != 4:
                            print(f"Warning: Skipping annotation in file {img_data['id']} due to incorrect number of points: {p}")
                            continue
                        f.write(f"{translate_ids[class_id]} {p[0]/normalize[0]} {p[1]/normalize[1]} {p[2]/normalize[0]} {p[3]/normalize[1]}\n")
                    elif ann['type'] == "polygon":
                        p = ann['points']
                        # reshape to N, 2 with every two consecutive values grouped in last dimension
                        points = np.array([p[::2], p[1::2]]).T
                        points[:, ::2] = points[:, ::2] / normalize[0]
                        points[:, 1::2] = points[:, 1::2] / normalize[1]
                        if id_to_name[translate_ids[ann['label_id']]] in ["CenterCircle"]:  # Ellipses
                            if ellipse_model.estimate(points):
                                cx, cy, a, b, theta = ellipse_model.params
                                f.write(f"{translate_ids[class_id]} {cx} {cy} {np.log(a)} {np.log(b)} {np.cos(2 * theta)} {np.sin(2 * theta)}\n")
        print("Conversion finished successfully!")

        if data['items'][0]['id'].count("/") > 0: # Check if datumaro exported with subfolders (e.g. "train/default/faiss/img1.jpg", etc.)
            folder_path = os.path.join(img_dir, "/".join(data['items'][0]['id'].split("/")[:-1])) # Fix for datumaro exporting with subfolders (e.g. "train/default/faiss/img1.jpg", etc.)
        else:
            folder_path = img_dir

        return folder_path, len(id_to_name), list(id_to_name.values()), special_classes


def datumaro_to_ultralyticsOBB(datumaro_dir: str, class_id_blacklist: Optional[list[int]] = None, merge_ids: Optional[dict[int, int]] = None,
                            permute_ids: Optional[list[int]] = None) -> Tuple[str, int, list, dict]:
    """
    Convert Image annotations exported from cvat in datumaro JSON format into DevilsYolo format.
    Lines are represented by their two endpoint coordinates in xyxy format.
    All coordinates are normalized
    Returns output path, number of classes, class_names and special_classes dict (for create dataset)

    Args:
        :param datumaro_dir: Path to the datumaro dataset root folder (the one that contains the annotations folder and the images folder)
        :param class_id_blacklist: List of class ids to ignore during conversion
        :param merge_ids: A dict of {x: int, y: int} pairs, where x is the class ID to be merged into class ID y (only class ID y is kept afterward)
        :param permute_ids: A list of indices used to permute the class IDs. The index of the list represents the old class ID, the value at that index represents the new class ID. This is applied after merging and blacklisting operations.

    Returns:
        path: Path to images and labels
        nc: Number of classes
        class_names: List of class names
        special_classes: Dict of special classes (lines, ellipses)
    """

    print("=" * 60)
    print(f"Converting datumaro to DevilsYolo format")
    print(f"Dataset root path: {datumaro_dir}")
    print("=" * 60)

    with open(f"{datumaro_dir}/annotations/default.json", 'r') as f:
        data = json.load(f)
    print("JSON loaded successfully")

    img_dir = f"{datumaro_dir}/images/default"

    id_to_name = {i: label["name"] for i, label in enumerate(data["categories"]["label"]["labels"])}

    if merge_ids is None:
        merge_ids = {}
    if class_id_blacklist is None:
        class_id_blacklist = []

    # update merge and blacklist information into id_to_name to return the correct nc, class_names and special_classes
    for k, v in merge_ids.items():
        print(f"Merging class {id_to_name[k]} into class {id_to_name[v]}")
        id_to_name[k] = id_to_name[v]
    for class_id in class_id_blacklist:
        print(f"Removing {class_id} ({id_to_name[class_id]}) from id_to_name due to blacklist")
        del id_to_name[class_id]
    # Rebuilding keys to ensure continuous class IDs and no duplicate names
    unique_names = list(id_to_name.fromkeys(id_to_name.values()))
    if permute_ids is None or len(permute_ids) == 0:
        permute_ids = range(len(unique_names))
    else:
        assert len(permute_ids) == len(
            unique_names), f"Error: number of permute_ids ({len(permute_ids)}) does not match number of unique class names ({len(unique_names)}) after merging and blacklisting operations."
    unique_names[:] = [unique_names[i] for i in permute_ids or range(len(unique_names))]
    new_id_to_name = {k: v for k, v in enumerate(unique_names)}  # Only used to generate information for building the dataset.yaml file
    name_to_new_id = {v: k for k, v in new_id_to_name.items()}
    translate_ids = {k: name_to_new_id[v] for k, v in id_to_name.items()}  # Usage: translate_ids[old_id] = new_id
    id_to_name = new_id_to_name
    print(f"id_to_name: {id_to_name}")
    print("Writing .txt annotations...")
    for img_ind, img_data in enumerate(tqdm(data['items'])):
        with (open(os.path.join(img_dir, img_data['id'] + ".txt"), 'w') as f):
            normalize = img_data["image"]["size"]
            normalize.reverse()  # datumaro saves height, width
            for ann_ind, ann in enumerate(img_data['annotations']):
                class_id = ann['label_id']
                class_id = merge_ids.get(class_id, class_id)
                if class_id in class_id_blacklist:
                    continue
                if ann['type'] == "bbox":  # BoundingBoxes
                    x1 = max(ann['bbox'][0] / normalize[0], 0.0)
                    y1 = max(ann['bbox'][1] / normalize[1], 0.0)
                    x2 = min((ann['bbox'][0] + ann['bbox'][2]) / normalize[0], 1.0)
                    y2 = min((ann['bbox'][1] + ann['bbox'][3]) / normalize[1], 1.0)
                    f.write(f"{translate_ids[class_id]} {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}\n")
                elif ann['type'] == "polygon":
                    p = ann['points']
                    # reshape to N, 2 with every two consecutive values grouped in last dimension
                    points = np.array([p[::2], p[1::2]]).T
                    points[:, ::2] = points[:, ::2] / normalize[0]
                    points[:, 1::2] = points[:, 1::2] / normalize[1]
                    if id_to_name[translate_ids[ann['label_id']]] in ["GoalPosts"]:  # GoalPosts
                        center = points.mean(axis=0)
                        centered_points = points - center
                        pca = PCA(n_components=2)
                        pca.fit(centered_points)
                        hauptachse_richtung = pca.components_[0]  # Erste Hauptachse (längste Achse)
                        # nebenachse_richtung = pca.components_[1]  # Zweite Hauptachse
                        angle = np.degrees(np.arctan2(*hauptachse_richtung)) % 360  # deliberately filled y, x params "incorrectly".

                        # Rotate points around their center to align hauptachse
                        rotated_points = _rotate_points_by_angle(centered_points, angle)
                        shifted_points = rotated_points + center

                        minx, maxx = min(shifted_points[:, 0]), max(shifted_points[:, 0])
                        miny, maxy = min(shifted_points[:, 1]), max(shifted_points[:, 1])

                        bb = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])
                        obb = _rotate_points_by_angle(bb, -angle, center)
                        obb = np.clip(obb, 0, 1)
                        f.write(f"{translate_ids[class_id]} {obb[0]} {obb[1]} {obb[2]} {obb[3]} {obb[4]} {obb[5]} {obb[6]} {obb[7]}\n")
    print("Conversion finished successfully!")

    if data['items'][0]['id'].count("/") > 0:  # Check if datumaro exported with subfolders (e.g. "train/default/faiss/img1.jpg", etc.)
        folder_path = os.path.join(img_dir, "/".join(
            data['items'][0]['id'].split("/")[:-1]))  # Fix for datumaro exporting with subfolders (e.g. "train/default/faiss/img1.jpg", etc.)
    else:
        folder_path = img_dir

    return folder_path, len(id_to_name), list(id_to_name.values()), {}


def _rotate_points_by_angle(points, angle, center=None):
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array(((c, -s), (s, c)))
    if center is None:
        return np.dot(points, rotation_matrix.T)
    else:
        return np.dot(points-center, rotation_matrix.T) + center