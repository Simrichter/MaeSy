import ast
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image
from typing import Optional

import numpy as np
import yaml
from skimage.measure import EllipseModel
from sklearn.decomposition import PCA
from tqdm import tqdm


def _safe_relative_path(path_value: str | Path) -> Path:
    normalized = str(path_value).replace("\\", "/")
    relative = Path(normalized)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or any(":" in part for part in relative.parts)
    ):
        raise ValueError(f"Expected a safe relative path, got {path_value!r}")
    return relative


def _datumaro_item_path(labels_dir: Path, item_id: str) -> Path:
    return labels_dir / _safe_relative_path(item_id).with_suffix(".txt")


def _is_finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _build_class_id_mapping(
    class_names: list[str],
    class_id_blacklist: Optional[list[int]],
    merge_ids: Optional[dict[int, int]],
    permute_ids: Optional[list[int]],
    special_class_ids: tuple[int, ...] = (),
) -> tuple[dict[int, Optional[int]], list[str]]:
    """Build one consistent source-ID to output-ID mapping."""
    class_count = len(class_names)
    blacklist = set(class_id_blacklist or [])
    merges = merge_ids or {}
    requested_ids = blacklist | set(merges) | set(merges.values())
    invalid_ids = [
        class_id for class_id in requested_ids
        if not isinstance(class_id, int) or class_id < 0 or class_id >= class_count
    ]
    if invalid_ids:
        raise ValueError(f"Class IDs must be in [0, {class_count}); got {invalid_ids}")

    for source_id, target_id in merges.items():
        if source_id in special_class_ids or target_id in special_class_ids:
            if source_id != target_id:
                raise ValueError("Geometry classes cannot be merged with other classes")

    def resolve(source_id: int) -> Optional[int]:
        current = source_id
        visited = set()
        while current in merges:
            if current in visited:
                raise ValueError(f"Class ID merge cycle includes class {current}")
            if current in blacklist:
                return None
            visited.add(current)
            current = merges[current]
        return None if current in blacklist else current

    destinations = sorted({
        destination for class_id in range(class_count)
        if (destination := resolve(class_id)) is not None
    })
    if permute_ids is not None and len(permute_ids) > 0:
        if sorted(permute_ids) != list(range(len(destinations))):
            raise ValueError(
                "permute_ids must be a permutation of the class IDs remaining "
                f"after merging and blacklisting (0..{len(destinations) - 1})"
            )
        # The public contract is old ID -> new ID, so build the inverse output order.
        order = [0] * len(destinations)
        for old_position, new_position in enumerate(permute_ids):
            order[new_position] = destinations[old_position]
        destinations = order

    final_ids = {destination: new_id for new_id, destination in enumerate(destinations)}
    old_to_new = {
        class_id: (
            None if (destination := resolve(class_id)) is None
            else final_ids[destination]
        )
        for class_id in range(class_count)
    }
    return old_to_new, [class_names[destination] for destination in destinations]


def _validate_new_output(output_path: str | Path, source_root: Path) -> Path:
    output_root = Path(output_path)
    resolved_out = output_root.resolve()
    resolved_source = source_root.resolve()
    if (
        resolved_out == resolved_source
        or resolved_source in resolved_out.parents
        or resolved_out in resolved_source.parents
    ):
        raise ValueError("Conversion output must not contain or be contained by the source directory")
    if output_root.exists():
        raise FileExistsError(f"Conversion output already exists: {output_root}")
    return output_root


def _validate_staging_location(output_root: Path, source_root: Path) -> None:
    resolved_output_parent = output_root.parent.resolve()
    resolved_source = source_root.resolve()
    if resolved_output_parent == resolved_source:
        raise ValueError("Conversion staging directory must be outside the source directory")


def _create_staging_output(output_root: Path) -> Path:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))


def _publish_staged_output(staging_root: Path, output_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(f"Conversion output already exists: {output_root}")
    os.replace(staging_root, output_root)


def _validate_datumaro_items(data: dict, source_img_dir: Path) -> list[tuple[dict, Path]]:
    items = []
    seen = set()
    for item in data.get("items", []):
        relative_image = _safe_relative_path(item["id"])
        relative_label = relative_image.with_suffix(".txt")
        if relative_label in seen:
            raise ValueError(f"Multiple Datumaro items map to label path {relative_label}")
        seen.add(relative_label)
        source_image = source_img_dir / relative_image
        output_label = _datumaro_item_path(Path("labels"), str(relative_image))
        if not output_label.is_relative_to(Path("labels")):
            raise ValueError(f"Datumaro item ID maps outside the label directory: {item['id']!r}")
        if not source_image.is_file():
            raise FileNotFoundError(f"Datumaro image {source_image} is missing")
        if not source_image.resolve().is_relative_to(source_img_dir.resolve()):
            raise ValueError(f"Datumaro image path escapes the image directory: {relative_image}")
        items.append((item, relative_image))
    return items


def _validate_datumaro_image_sizes(
    items: list[tuple[dict, Path]], source_img_dir: Path
) -> None:
    for item, relative_image in items:
        size = item.get("image", {}).get("size")
        if (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or not all(_is_finite_number(value) and value > 0 and int(value) == value for value in size)
        ):
            raise ValueError(f"Invalid Datumaro image size for {item.get('id')!r}: {size!r}")
        with Image.open(source_img_dir / relative_image) as image:
            if image.size != (size[1], size[0]):
                raise ValueError(
                    f"Datumaro image size for {item['id']!r} is {size}, "
                    f"but the source image is {image.size[::-1]}"
                )


def _write_datumaro_yolo(
    datumaro_dir: str | Path,
    output_path: str | Path,
    box_format: str,
    class_id_blacklist: Optional[list[int]],
    merge_ids: Optional[dict[int, int]],
    permute_ids: Optional[list[int]],
) -> tuple[Path, list[str]]:
    if box_format not in {"xyxy", "cxcywh"}:
        raise ValueError("box_format must be 'xyxy' or 'cxcywh'")
    datumaro_root = Path(datumaro_dir)
    source_img_dir = datumaro_root / "images" / "default"
    annotation_path = datumaro_root / "annotations" / "default.json"
    with annotation_path.open(encoding="utf-8") as source:
        data = json.load(source)

    source_class_names = [
        label["name"] for label in data["categories"]["label"]["labels"]
    ]
    special_source_ids = tuple(
        index for index, name in enumerate(source_class_names)
        if name.lower() in {"line", "lines", "centercircle", "ellipse", "ellipses"}
    )
    old_to_new, class_names = _build_class_id_mapping(
        source_class_names, class_id_blacklist, merge_ids, permute_ids,
        special_class_ids=special_source_ids,
    )
    items = _validate_datumaro_items(data, source_img_dir)
    _validate_datumaro_image_sizes(items, source_img_dir)
    for item, _relative_image in items:
        height, width = item["image"]["size"]
        for annotation in item.get("annotations", []):
            label_id = annotation.get("label_id", -1)
            if not isinstance(label_id, int) or isinstance(label_id, bool) or label_id < 0 or label_id >= len(old_to_new):
                raise ValueError(f"Invalid Datumaro class ID {label_id}")
            kind = annotation.get("type")
            if kind == "bbox":
                bbox = annotation.get("bbox", [])
                if (
                    len(bbox) != 4
                    or not all(_is_finite_number(value) for value in bbox)
                    or bbox[2] < 0
                    or bbox[3] < 0
                ):
                    raise ValueError(f"Invalid Datumaro bounding box in {item['id']!r}")
            elif kind == "polyline":
                points = annotation.get("points", [])
                if (
                    len(points) < 4
                    or len(points) % 2
                    or not all(_is_finite_number(value) for value in points)
                ):
                    raise ValueError(f"Invalid Datumaro polyline in {item['id']!r}")
            elif kind == "polygon":
                if old_to_new[label_id] is None:
                    continue
                points = annotation.get("points", [])
                if (
                    len(points) < 10
                    or len(points) % 2
                    or not all(_is_finite_number(value) for value in points)
                ):
                    raise ValueError(f"Invalid Datumaro polygon in {item['id']!r}")
                source_name = source_class_names[label_id].lower()
                if source_name not in {"centercircle", "ellipse", "ellipses"}:
                    raise ValueError(
                        f"Standard Maesy conversion does not support polygon class {source_class_names[label_id]!r}"
                    )
            else:
                raise ValueError(f"Unsupported Datumaro annotation type {kind!r}")

    output_root = _validate_new_output(output_path, datumaro_root)
    _validate_staging_location(output_root, datumaro_root)
    staging_root = _create_staging_output(output_root)
    output_images = staging_root / "default" / "images"
    output_labels = staging_root / "default" / "labels"

    try:
        ellipse_model = EllipseModel()
        for item, relative_image in tqdm(items):
            output_image = output_images / relative_image
            output_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_img_dir / relative_image, output_image)
            output_label = output_labels / relative_image.with_suffix(".txt")
            output_label.parent.mkdir(parents=True, exist_ok=True)
            height, width = item["image"]["size"]
            with output_label.open("w", encoding="utf-8") as labels:
                for annotation in item.get("annotations", []):
                    old_id = annotation["label_id"]
                    if old_id < 0 or old_id >= len(old_to_new):
                        raise ValueError(f"Invalid Datumaro class ID {old_id}")
                    class_id = old_to_new[old_id]
                    if class_id is None:
                        continue
                    kind = annotation["type"]
                    if kind == "bbox":
                        x, y, box_width, box_height = annotation["bbox"]
                        x1 = min(max(x / width, 0.0), 1.0)
                        y1 = min(max(y / height, 0.0), 1.0)
                        x2 = min(max((x + box_width) / width, 0.0), 1.0)
                        y2 = min(max((y + box_height) / height, 0.0), 1.0)
                        if box_format == "xyxy":
                            coordinates = (x1, y1, x2, y2)
                        else:
                            coordinates = (
                                (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                            )
                        labels.write(f"{class_id} " + " ".join(map(str, coordinates)) + "\n")
                    elif kind == "polyline":
                        points = annotation["points"]
                        for index in range(0, len(points) - 2, 2):
                            x1, y1, x2, y2 = points[index:index + 4]
                            labels.write(
                                f"{class_id} {x1 / width} {y1 / height} {x2 / width} {y2 / height}\n"
                            )
                    elif kind == "polygon":
                        name = class_names[class_id]
                        flat = annotation["points"]
                        points = np.asarray(list(zip(flat[::2], flat[1::2])), dtype=float)
                        points[:, 0] /= width
                        points[:, 1] /= height
                        if ellipse_model.estimate(points):
                            cx, cy, axis_a, axis_b, theta = ellipse_model.params
                            labels.write(
                                f"{class_id} {cx} {cy} {np.log(axis_a)} {np.log(axis_b)} "
                                f"{np.cos(2 * theta)} {np.sin(2 * theta)}\n"
                            )
                    else:
                        raise ValueError(f"Unsupported Datumaro annotation type {kind!r}")

        special_classes = {}
        for name in class_names:
            lowered = name.lower()
            if lowered in {"line", "lines"}:
                special_classes["lines"] = name
            elif lowered in {"centercircle", "ellipse", "ellipses"}:
                special_classes["ellipses"] = name
        (staging_root / "dataset.yaml").write_text(
            yaml.safe_dump({
                "path": str(output_root.resolve()),
                "nc": len(class_names),
                "names": class_names,
                "box_format": box_format,
                "default": "default",
                **special_classes,
            }),
            encoding="utf-8",
        )
        _publish_staged_output(staging_root, output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return output_root, class_names


def robert_to_devils_yolo(
    dataset_dir: str | Path,
    box_format: str = "xyxy",
    class_id_blacklist: Optional[list[int]] = None,
    merge_ids: Optional[dict[int, int]] = None,
    permute_ids: Optional[list[int]] = None,
    output_path: str | Path | None = None,
):
    """Convert a Robert simulator dataset into a non-destructive Maesy dataset."""
    if box_format not in {"xyxy", "cxcywh"}:
        raise ValueError("box_format must be 'xyxy' or 'cxcywh'")
    if output_path is None:
        raise ValueError("An output_path is required to avoid modifying the source dataset")
    dataset_root = Path(dataset_dir)
    source_class_names = [
        "Trionda Ball 2026(Clone)", "K1(Clone)", "PenaltyCross", "Line", "CenterCircle"
    ]
    old_to_new, class_names = _build_class_id_mapping(
        source_class_names, class_id_blacklist, merge_ids, permute_ids,
        special_class_ids=(3, 4),
    )
    splits = sorted(
        path.name for path in dataset_root.iterdir()
        if path.is_dir() and path.name in {"train", "val", "test"}
    )
    if not splits:
        raise ValueError(f"No train/val/test splits found in {dataset_root}")

    # Validate all source images and annotations before creating destination files.
    # Missing labels represent background examples and become empty label files.
    split_data = []
    for split in splits:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        if not images_dir.is_dir():
            raise ValueError(f"Could not find '{split}/images' directory")
        if not labels_dir.is_dir():
            labels_dir = None
        image_paths = sorted(
            path for path in images_dir.rglob("*")
            if path.is_file()
            and path.resolve().is_relative_to(images_dir.resolve())
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as error:
                raise ValueError(f"Could not read Robert image {image_path}: {error}") from error
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions for {image_path}")
        image_keys = [
            path.relative_to(images_dir).with_suffix("").as_posix().casefold()
            for path in image_paths
        ]
        if len(set(image_keys)) != len(image_keys):
            raise ValueError(f"Multiple Robert images share the same relative stem in {images_dir}")
        label_by_stem = {}
        if labels_dir is not None:
            for path in labels_dir.rglob("*"):
                if (
                    not path.is_file()
                    or not path.resolve().is_relative_to(labels_dir.resolve())
                    or path.suffix.lower() != ".txt"
                ):
                    continue
                key = path.relative_to(labels_dir).with_suffix("").as_posix().casefold()
                if key in label_by_stem:
                    raise ValueError(f"Multiple Robert labels map to image {key!r}")
                label_by_stem[key] = path
        for image_path in image_paths:
            relative = image_path.relative_to(images_dir)
            label_path = label_by_stem.get(relative.with_suffix("").as_posix().casefold())
            split_data.append((split, label_path, relative, image_path))
        image_keys = {
            image.relative_to(images_dir).with_suffix("").as_posix().casefold()
            for image in image_paths
        }
        for label_key, label_path in label_by_stem.items():
            if label_key not in image_keys:
                raise FileNotFoundError(f"No image matches Robert label {label_path}")

    prepared_data = []
    for split, label_path, relative, image_path in split_data:
        output_lines = []
        category = None
        source_lines = label_path.read_text(encoding="utf-8").splitlines() if label_path else []
        for raw in source_lines:
            line = raw.strip()
            section = line.removesuffix(":")
            if section in {"BoundingBoxes", "GoalPosts", "CenterCircle", "PenaltyPoints", "Lines"}:
                category = section
                continue
            if not line:
                continue
            if category == "BoundingBoxes":
                if ":" not in line:
                    raise ValueError(f"Invalid bounding box in {label_path}: {raw}")
                class_name, coordinates = line.split(":", 1)
                try:
                    source_id = source_class_names.index(class_name.strip())
                except ValueError as error:
                    raise ValueError(f"Unknown Robert class {class_name.strip()!r} in {label_path}") from error
                if old_to_new[source_id] is None:
                    continue
                try:
                    values = [float(value) for value in coordinates.split()]
                except ValueError as error:
                    raise ValueError(f"Invalid bounding box in {label_path}: {raw}") from error
                if len(values) != 4 or not all(_is_finite_number(value) for value in values):
                    raise ValueError(f"Invalid bounding box in {label_path}: {raw}")
                if values[2] < 0 or values[3] < 0:
                    raise ValueError(f"Invalid bounding-box dimensions in {label_path}: {raw}")
                cx, cy, width, height = values
                coords = (
                    (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
                    if box_format == "xyxy" else (cx, cy, width, height)
                )
                output_lines.append(f"{old_to_new[source_id]} " + " ".join(map(str, coords)))
            elif category == "PenaltyPoints":
                source_id = 2
                if old_to_new[source_id] is None:
                    continue
                try:
                    values = [float(value) for value in line.split()]
                except ValueError as error:
                    raise ValueError(f"Invalid penalty point in {label_path}: {raw}") from error
                if len(values) != 4 or not all(_is_finite_number(value) for value in values):
                    raise ValueError(f"Invalid penalty point in {label_path}: {raw}")
                if values[2] < 0 or values[3] < 0:
                    raise ValueError(f"Invalid penalty-point dimensions in {label_path}: {raw}")
                cx, cy, width, height = values
                coords = (
                    (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
                    if box_format == "xyxy" else (cx, cy, width, height)
                )
                output_lines.append(f"{old_to_new[source_id]} " + " ".join(map(str, coords)))
            elif category == "Lines":
                source_id = 3
                if old_to_new[source_id] is None:
                    continue
                try:
                    points = [float(value.strip(" ()")) for value in line.split(",")]
                except ValueError as error:
                    raise ValueError(f"Invalid line annotation in {label_path}: {raw}") from error
                if len(points) != 4 or not all(_is_finite_number(value) for value in points):
                    raise ValueError(f"Invalid line annotation in {label_path}: {raw}")
                x1, y1, x2, y2 = points
                output_lines.append(
                    f"{old_to_new[source_id]} {x1 / 544} {1 - y1 / 448} {x2 / 544} {1 - y2 / 448}"
                )
            elif category == "CenterCircle":
                source_id = 4
                if old_to_new[source_id] is None:
                    continue
                try:
                    points = np.asarray(ast.literal_eval(f"[{line}]"), dtype=float)
                except (SyntaxError, ValueError, TypeError) as error:
                    raise ValueError(f"Invalid center-circle annotation in {label_path}: {raw}") from error
                if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5 or not np.isfinite(points).all():
                    raise ValueError(f"Invalid center-circle annotation in {label_path}: {raw}")
                points[:, 0] /= 544
                points[:, 1] = 1 - points[:, 1] / 448
                model = EllipseModel()
                if model.estimate(points):
                    cx, cy, axis_a, axis_b, theta = model.params
                    if not all(_is_finite_number(value) and value > 0 for value in (axis_a, axis_b)):
                        raise ValueError(f"Invalid fitted center circle in {label_path}: {raw}")
                    output_lines.append(
                        f"{old_to_new[source_id]} {cx} {cy} {np.log(axis_a)} {np.log(axis_b)} "
                        f"{np.cos(2 * theta)} {np.sin(2 * theta)}"
                    )
        prepared_data.append((split, label_path, relative, image_path, output_lines))

    output_root = _validate_new_output(output_path, dataset_root)
    _validate_staging_location(output_root, dataset_root)
    staging_root = _create_staging_output(output_root)
    special_classes = {}
    for source_id, source_name in ((3, "Line"), (4, "CenterCircle")):
        output_id = old_to_new[source_id]
        if output_id is not None:
            key = "lines" if source_id == 3 else "ellipses"
            special_classes[key] = class_names[output_id]

    try:
        for split, label_path, relative, image_path, output_lines in tqdm(prepared_data):
            output_image = staging_root / split / "images" / relative
            output_label = staging_root / split / "labels" / relative.with_suffix(".txt")
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, output_image)
            output_label.write_text(
                "\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8"
            )

        (staging_root / "dataset.yaml").write_text(
            yaml.safe_dump({
                "path": str(output_root.resolve()),
                "nc": len(class_names),
                "names": class_names,
                "box_format": box_format,
                **{split: split for split in splits},
                **special_classes,
            }),
            encoding="utf-8",
        )
        _publish_staged_output(staging_root, output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return output_root, len(class_names), class_names, special_classes


def datumaro_to_devils_yolo(
    datumaro_dir: str | Path,
    box_format: str = "xyxy",
    class_id_blacklist: Optional[list[int]] = None,
    merge_ids: Optional[dict[int, int]] = None,
    permute_ids: Optional[list[int]] = None,
    output_path: str | Path | None = None,
) -> tuple[Path, int, list[str], dict[str, str]]:
    """Convert Datumaro labels and copied images to the Maesy detection layout."""
    if output_path is None:
        raise ValueError("An output_path is required to avoid modifying the source dataset")
    output_root, class_names = _write_datumaro_yolo(
        datumaro_dir, output_path, box_format, class_id_blacklist, merge_ids, permute_ids
    )
    with (output_root / "dataset.yaml").open(encoding="utf-8") as config_file:
        special_classes = {
            key: value for key, value in yaml.safe_load(config_file).items()
            if key in {"lines", "ellipses"}
        }
    return output_root, len(class_names), class_names, special_classes


def datumaro_to_ultralyticsOBB(
    datumaro_dir: str | Path,
    box_format: str = "xyxy",
    class_id_blacklist: Optional[list[int]] = None,
    merge_ids: Optional[dict[int, int]] = None,
    permute_ids: Optional[list[int]] = None,
    output_path: str | Path | None = None,
) -> tuple[Path, int, list[str], dict]:
    """Export Datumaro polygons as Ultralytics OBB labels (not Maesy detection data)."""
    if box_format not in {"xyxy", "cxcywh"}:
        raise ValueError("box_format must be 'xyxy' or 'cxcywh'")
    if output_path is None:
        raise ValueError("An output_path is required for OBB export")

    datumaro_root = Path(datumaro_dir)
    annotation_path = datumaro_root / "annotations" / "default.json"
    source_img_dir = datumaro_root / "images" / "default"
    with annotation_path.open(encoding="utf-8") as source:
        data = json.load(source)
    source_class_names = [
        label["name"] for label in data["categories"]["label"]["labels"]
    ]
    old_to_new, class_names = _build_class_id_mapping(
        source_class_names, class_id_blacklist, merge_ids, permute_ids
    )
    items = _validate_datumaro_items(data, source_img_dir)
    _validate_datumaro_image_sizes(items, source_img_dir)
    for item, _relative_image in items:
        height, width = item["image"]["size"]
        if (
            not _is_finite_number(height)
            or not _is_finite_number(width)
            or height <= 0
            or width <= 0
        ):
            raise ValueError(f"Invalid Datumaro image size {item['id']!r}: {item['image']['size']}")
        for annotation in item.get("annotations", []):
            label_id = annotation.get("label_id", -1)
            if not isinstance(label_id, int) or isinstance(label_id, bool) or label_id < 0 or label_id >= len(old_to_new):
                raise ValueError(f"Invalid Datumaro class ID {label_id}")
            if annotation["type"] == "polygon":
                points = annotation.get("points", [])
                if len(points) != 8:
                    raise ValueError("Ultralytics OBB polygons must have exactly four corners")
                if not all(_is_finite_number(coordinate) for coordinate in points):
                    raise ValueError("Datumaro OBB polygon coordinates must be finite numbers")
            elif annotation["type"] == "bbox":
                bbox = annotation.get("bbox", [])
                if (
                    len(bbox) != 4
                    or not all(_is_finite_number(value) for value in bbox)
                ):
                    raise ValueError("Datumaro OBB bounding boxes must contain four finite numbers")
                x, y, box_width, box_height = bbox
                if box_width < 0 or box_height < 0:
                    raise ValueError("Datumaro OBB bounding boxes must have non-negative dimensions")
            else:
                raise ValueError(f"OBB conversion does not support {annotation['type']!r}")
    output_root = _validate_new_output(output_path, datumaro_root)
    _validate_staging_location(output_root, datumaro_root)
    staging_root = _create_staging_output(output_root)
    output_images = staging_root / "images" / "default"
    output_labels = staging_root / "labels" / "default"
    try:
        for item, relative_image in tqdm(items):
            source_image = source_img_dir / relative_image
            output_image = output_images / relative_image
            output_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, output_image)
            output_label = output_labels / relative_image.with_suffix(".txt")
            output_label.parent.mkdir(parents=True, exist_ok=True)
            height, width = item["image"]["size"]
            rows = []
            for annotation in item.get("annotations", []):
                old_id = annotation["label_id"]
                if old_id < 0 or old_id >= len(old_to_new):
                    raise ValueError(f"Invalid Datumaro class ID {old_id}")
                class_id = old_to_new[old_id]
                if class_id is None:
                    continue
                if annotation["type"] == "bbox":
                    x, y, box_width, box_height = annotation["bbox"]
                    x1 = min(max(x / width, 0.0), 1.0)
                    y1 = min(max(y / height, 0.0), 1.0)
                    x2 = min(max((x + box_width) / width, 0.0), 1.0)
                    y2 = min(max((y + box_height) / height, 0.0), 1.0)
                    points = (x1, y1, x2, y1, x2, y2, x1, y2)
                elif annotation["type"] == "polygon":
                    flat = annotation["points"]
                    points = tuple(
                        coordinate
                        for x, y in zip(flat[::2], flat[1::2])
                        for coordinate in (x / width, y / height)
                    )
                else:
                    raise ValueError(f"OBB conversion does not support {annotation['type']!r}")
                rows.append(f"{class_id} " + " ".join(map(str, points)))
            output_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

        (staging_root / "dataset.yaml").write_text(
            yaml.safe_dump({
                "path": str(output_root.resolve()),
                "train": "images/default",
                "nc": len(class_names),
                "names": class_names,
            }),
            encoding="utf-8",
        )
        _publish_staged_output(staging_root, output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return output_root, len(class_names), class_names, {}


def _rotate_points_by_angle(points, angle, center=None):
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    if center is None:
        return np.dot(points, rotation.T)
    return np.dot(points - center, rotation.T) + center
