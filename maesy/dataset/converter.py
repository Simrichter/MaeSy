import ast
import json
import os
from pathlib import Path

import numpy as np
from skimage.measure import EllipseModel
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

def robert_to_devils_yolo(labels_path: str | Path, out_path: str | Path = ""):
    """
        Convert Image annotations from Robert's Unity simulator outputs to devils_yolo format.

        Args:
            :param labels_path: Path to the labels folder (only .txt is considered)
            :param out_path: Path to the output folder (if not specified, a subfolder "DevilsYolo" is created)
    """

    name_to_id = {
        "Trionda Ball 2026(Clone)": 0,
        "K1(Clone)": 1,
        "PenaltyCross": 2,
        "Line": 3,
        "CenterCircle": 4,
    }

    print("="*60)
    print(f"Converting Robert's Unity Simulator annotations to DevilsYolo format")
    print(f"Labels path: {labels_path}")
    print("=" * 60)


    labels_path: Path = Path(labels_path)
    splits = [f for f in os.listdir(labels_path) if f in ["train", "val", "test"]]
    print(f"Found {len(splits)} splits: {splits}")
    for split in splits:
        print(f"Handling {split}...")
        path = labels_path/split/"labels"
        if not os.path.exists(path):
            raise ValueError(f"Could not find '{split}/labels' Folder in dataset root!!")
        out_path: Path = Path(path)/"DevilsYolo" if out_path == "" else Path(out_path)
        out_path.mkdir(exist_ok=True)

        txts = [t for t in os.listdir(path) if t.endswith(".txt")]
        model = EllipseModel()
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
                        class_id, coords = line.strip().split(":")
                        parts = coords.strip().split()
                        if len(parts) != 4:
                            print(f"Skipping BoundingBox line in file {label_file} due to incorrect format: {line}")
                            continue
                        cx, cy, w, h = parts
                        cx, cy, w, h = float(cx), float(cy), float(w), float(h) # Explicit cast to ensure datatype compatibility
                        new_lines.append(f"{name_to_id[class_id]} {cx} {cy} {w} {h}\n")
                    case "Lines":
                        line_parts = line.strip().split(", ")
                        if len(line_parts) == 4: # Subtype: Line
                            x1, y1, x2, y2 = [float(lp.lstrip("(").rstrip("),")) for lp in line_parts]
                            x1, x2 = x1/544, x2/544
                            y1, y2 = 1-y1/448, 1-y2/448 # TODO: Quickfix for horizontally flipped lines????
                            new_lines.append(f"{name_to_id['Line']} {x1} {y1} {x2} {y2}\n") # Class ID 1 for lines, can be changed if needed
                        else: # Subtype: CornerArc
                            ... # TODO
                    case "PenaltyPoints":
                        parts = line.strip().split()
                        if len(parts) != 4:
                            print(f"Skipping PenaltyPoint line in file {label_file} due to incorrect format: {line}")
                            continue
                        cx, cy, w, h = parts
                        cx, cy, w, h = float(cx), float(cy), float(w), float(h)  # Explicit cast to ensure datatype compatibility
                        new_lines.append(f"{name_to_id['PenaltyCross']} {cx} {cy} {w} {h}\n")
                    case "CenterCircle":
                        points = np.array(ast.literal_eval(f"[{line}]"))
                        points[:,::2] = points[:,::2]/544
                        points[:,1::2] = 1-points[:,1::2]/448
                        if model.estimate(points):
                            cx, cy, a, b, theta = model.params
                            # Convert to cholesky representation
                            cx, cy, l11, l12, l22 = ellipse_to_cholesky(cx, cy, a, b, theta)
                            new_lines.append(f"{name_to_id['CenterCircle']} {cx} {cy} {l11} {l12} {l22}\n")
                        else:
                            print(f"Failed to estimate ellipse from line '{line}' in file {label_file}")
                    case _:
                        # ignore other categories for now
                        ...

            with open(os.path.join(out_path, label_file), 'w') as f:
                f.writelines(new_lines)


def datumaro_to_devils_yolo(datumaro_dir: str):
        """
        Convert Image annotations exported from cvat in datumaro JSON format into DevilsYolo format.
        Lines are represented by their two endpoint coordinates in xyxy format.
        All coordinates are normalized

        Args:
            :param datumaro_dir: Path to the datumaro dataset root folder (the one that contains the annotations folder and the images folder)
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
                                    print(
                                        f"Skipping double point for annotation {ann_i} in file {img_dat['id']} ({img_i}) due to zero-length vector: {p1} and {p2}")

                            annot['points'] = anns_to_be_added[0]['points']
                            anns_to_be_added = anns_to_be_added[1:]
                img_dat['annotations'].extend(anns_to_be_added)

        with open(f"{datumaro_dir}/annotations/default.json", 'r') as f:
            data = json.load(f)
        print("JSON loaded successfully")
        print("Checking for multi-point line annotations and fixing them if necessary...")
        _fix_multipoint_line(data)

        img_dir = f"{datumaro_dir}/images/default"

        print("Writing .txt annotations...")
        for img_ind, img_data in enumerate(tqdm(data['items'])):
            with (open(os.path.join(img_dir, img_data['id'] + ".txt"), 'w') as f):
                normalize = img_data["image"]["size"]
                normalize.reverse() # datumaro saves height, width
                for ann_ind, ann in enumerate(img_data['annotations']):
                    if ann['type'] == "bbox":
                        cx = ann['bbox'][0] / normalize[0]
                        cy = ann['bbox'][1] / normalize[1]
                        w = ann['bbox'][2] / normalize[0]
                        h = ann['bbox'][3] / normalize[1]
                        cx += 0.5 * w
                        cy += 0.5 * h
                        f.write(f"{ann['label_id']} {cx} {cy} {w} {h}\n")
                    elif ann['type'] == "polyline":
                        p = ann['points']
                        if len(p) != 4:
                            print(
                                f"Warning: Skipping annotation in file {img_data['id']} due to incorrect number of points: {p}")
                            continue
                        f.write(f"{ann['label_id']} {p[0]/normalize[0]} {p[1]/normalize[1]} {p[2]/normalize[0]} {p[3]/normalize[1]}\n")
        print("Conversion finished successfully!")