"""Dataset manager for downloading and managing datasets."""

import math
import os
import shutil
import zipfile


import requests
from pathlib import Path
from typing import Optional, Dict, List

import yaml
from tqdm import tqdm
import tarfile
import random
import tempfile


def _allocate_split_counts(num_images: int, percentages: List[float]) -> List[int]:
    """Allocate every image once and populate nonzero splits when possible."""
    if num_images <= 0:
        return [0, 0, 0]

    exact_counts = [num_images * percentage for percentage in percentages]
    counts = [math.floor(value) for value in exact_counts]
    remaining = num_images - sum(counts)
    remainder_order = sorted(
        range(len(percentages)),
        key=lambda index: (exact_counts[index] - counts[index], percentages[index], -index),
        reverse=True,
    )
    for index in remainder_order[:remaining]:
        counts[index] += 1

    active_splits = [index for index, percentage in enumerate(percentages) if percentage > 0]
    target_nonempty = min(num_images, len(active_splits))
    while sum(counts[index] > 0 for index in active_splits) < target_nonempty:
        receivers = [index for index in remainder_order if index in active_splits and counts[index] == 0]
        donors = [index for index in active_splits if counts[index] > 1]
        if not receivers or not donors:
            break
        receiver = receivers[0]
        donor = max(
            donors,
            key=lambda index: (
                counts[index] - exact_counts[index], counts[index], -index
            ),
        )
        counts[donor] -= 1
        counts[receiver] += 1
    return counts


class DatasetManager:
    """Manages dataset downloading, extraction, and organization."""

    def __init__(self, data_root: str = "./data"):
        """
        Initialize DatasetManager.
        
        Args:
            data_root: Root directory for storing datasets
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def download_data(
            self,
            url: str,
            dataset_name: str,
            extract: bool = True,
            force: bool = False,
            keep_temp: bool = False
    ) -> Path:
        """
        Download dataset from URL.
        
        Args:
            url: URL to download dataset from
            dataset_name: Name of the dataset
            extract: Whether to extract if zip file
            force: Force re-download even if exists
            keep_temp: Whether to keep temp files used during downloading (i.e. zip folders)
            
        Returns:
            Path to downloaded/extracted dataset
        """
        dataset_dir = self.data_root / dataset_name

        if dataset_dir.exists() and not force:
            print(f"Dataset {dataset_name} already exists at {dataset_dir}")
            return dataset_dir

        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Download file
        filename = url.split("/")[-1]
        filepath = dataset_dir / filename

        print(f"Downloading {dataset_name} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Extract if zip file
        if extract:
            if filepath.suffix == '.zip':
                print(f"Extracting {filename}...")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                print("...Done")
            elif filepath.suffix == '.tar':
                print(f"Extracting {filename}...")
                tar = tarfile.open(filepath)
                tar.extractall(dataset_dir)
                tar.close()
                print("...Done")
            else:
                print(f"Warning: Failed to extract downloaded file. Filetype {filepath.suffix} not supported")
                return dataset_dir

            if not keep_temp:
                os.remove(filepath)

        return dataset_dir

    def cluster_data(self, folder_names: list[str], chosen_paths: list[str], cluster_method: str = "resnet_kmeans", step: int = 1, start_index: int = 0,
                     batch_size: int = 256, **kwargs) -> List[Path]:
        """
            Clusters all images in the given folders into num_clusters clusters using the specified method.
            The paths of the images to be used are returned as a list

        Arguments:
            :param folder_names: A list of paths to image data folders
            :param chosen_paths: List of already chosen image paths to include as initial representatives (pass None if not wanted)
            :param cluster_method: A string specifying the clustering method to use. Default is resnet_kmeans
                                   Options: "resnet_kmeans", "sequential_similarity"
            :param step: Step size for selecting images from folders. Default=1 (use all images)
            :param start_index: Start index for selecting images from folders. Default=0
            :param batch_size: The batch size to be used for feature extraction. Default=256
            :param kwargs: Additional specifications like 'num_clusters' or 'similarity_threshold'

            :return: A list of paths to the selected images
        """
        match cluster_method:
            case "resnet_faiss":
                from clusterdevil.clustering_methods import ResnetFaiss
                # return cluster(folder_names, chosen_paths, similarity_threshold, step=step, start_index=start_index)
                return ResnetFaiss().run(paths=folder_names, chosen_paths=chosen_paths, step=step, start_index=start_index, similarity_threshold=kwargs.get("similarity_threshold", 0.85), batch_size=batch_size)
                # TODO: Make threshold a param
            case "resnet_kmeans":
                from clusterdevil.clustering_methods.resnet_kmeans import ResnetKmeans
                return ResnetKmeans().run(paths=folder_names, chosen_paths=chosen_paths, step=step, start_index=start_index, num_clusters=kwargs.get("num_clusters", 200), batch_size=batch_size)
                # return cluster_with_kmeans(folder_names, n_c=num_clusters, step=step, start_index=start_index)
            case _:
                raise ValueError(f"Unknown clustering method {cluster_method}")

    @staticmethod
    def _validate_resize(resize: List[int] | None) -> None:
        if resize is None:
            return
        if (
            not isinstance(resize, (list, tuple))
            or not 1 <= len(resize) <= 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in resize)
        ):
            raise ValueError(
                "resize must be None or one/two positive integer dimensions: [WIDTH] or [WIDTH HEIGHT]"
            )

    @staticmethod
    def _copy_resize(
        source_paths: List[Path],
        target_paths: List[Path],
        resize: List[int] | None,
        label_target_paths: Optional[List[Path]] = None,
        require_labels: bool = False,
        source_label_paths: Optional[List[Path]] = None,
    ):
        DatasetManager._validate_resize(resize)
        if len(source_paths) != len(target_paths):
            raise ValueError("Each source image must have exactly one target path")
        if label_target_paths is not None and len(source_paths) != len(label_target_paths):
            raise ValueError("Each source image must have exactly one label target path")
        if source_label_paths is not None and len(source_paths) != len(source_label_paths):
            raise ValueError("Each source image must have exactly one source label path")
        from PIL import Image

        written_targets = []
        for index, (img_file, target_path) in enumerate(
            tqdm(zip(source_paths, target_paths), desc=f"Copying {len(source_paths)} files")
        ):
            if not img_file.is_file() or img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                written_targets.append(None)
                continue

            label_file = (
                source_label_paths[index]
                if source_label_paths is not None
                else img_file.with_suffix(".txt")
            )
            label_target = (
                label_target_paths[index] if label_target_paths is not None else None
            )
            if require_labels and not label_file.is_file():
                raise FileNotFoundError(
                    f"Label file {label_file} is missing for source image {img_file}"
                )

            target_path.parent.mkdir(parents=True, exist_ok=True)
            if label_target is not None:
                label_target.parent.mkdir(parents=True, exist_ok=True)

            # Rename the image and its label together to keep their stems paired.
            while target_path.exists() or (
                label_target is not None and label_target.exists()
            ):
                target_path = target_path.with_name(
                    f"{target_path.stem}_copy{target_path.suffix}"
                )
                if label_target is not None:
                    label_target = label_target.with_name(f"{target_path.stem}.txt")

            if resize is None:
                shutil.copy2(img_file, target_path)
            else:
                with Image.open(img_file) as image:
                    image.resize((resize[0], resize[1] if len(resize) == 2 else resize[0])).save(
                        target_path
                    )

            if label_target is not None:
                if label_file.is_file():
                    shutil.copy2(label_file, label_target)
                else:
                    label_target.touch()
            written_targets.append(target_path)
        return written_targets

    @staticmethod
    def _create_yaml(dataset_dir: Path, splits: List[str], num_classes: Optional[int] = None, class_names: Optional[List[str]] = None, special_classes: Optional[Dict[str, str]] = None, box_format: str = "xyxy", yaml_root: Optional[Path] = None):
        """
            Create a dataset.yaml file in MaesyDataset format.

            Args:
                :param dataset_dir: Root directory of the (newly) created dataset
                :param splits: List of strings specifying the existing splits
                :param num_classes: Number of classes in the dataset
                :param class_names: List of strings specifying the class names (ordered by class-id)
                :param special_classes: Dict of strings, where keys are from ["lines", "ellipses"] and values are from class_names
                :param box_format: Box format specifier string (xyxy, cxcywh, etc.)
        """
        if num_classes is None:
            if class_names is None:
                num_classes = 0
            else:
                num_classes = len(class_names)

        yaml_content = {
            'path' : str((yaml_root or dataset_dir).resolve()),
            'nc': num_classes,
            'names': class_names if class_names is not None else [f'class_{i}' for i in range(num_classes)],
            'box_format': box_format
        }
        for split in splits:
            yaml_content[split] = str(split)
        if special_classes is not None:
            yaml_content.update(special_classes)
        yaml_path = dataset_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)

    def create_dataset(self,
                       folder_names: list[str],
                       chosen_paths: list[str],
                       dataset_name: str,
                       split_percentages: list[float],
                       resize: list[int],
                       with_labels: bool,
                       step: int,
                       start_index: int,
                       del_folders: bool,  # TODO: Add functionality
                       cluster_method: str,
                       left_right: bool,
                       convert: Optional[str] = None,
                       convert_id_blacklist: Optional[List[int]] = None,
                       merge_ids: Optional[dict[int, int]] = None,
                       permute_ids: Optional[list[int]] = None,
                       nc: Optional[int] = None,
                       box_format: str = "xyxy",
                       similarity_threshold: Optional[float] = None,
                       num_clusters: Optional[int] = None,
                       cluster_batch_size: int = 256,
                       class_names: Optional[List[str]] = None,
                       special_classes: Optional[Dict[str, str]] = None,
                       ) -> Path:
        """
        Combines multiple folders with images into a single dataset

        Arguments:
            :param folder_names: List of folders that contain images
            :param chosen_paths: List of already chosen image paths to include as initial representatives (pass None if not wanted)
            :param dataset_name: Name of the dataset
            :param split_percentages: List of percentages for the data subsets in format [train, val, test]. Defaults to [0.8, 0.1, 0.1] if not/incorrectly specified
            :param resize: Resize images to WIDTH HEIGHT or WIDTH² if HEIGHT not specified
            :param with_labels: Whether to include label files in coco style (.txt files with same name as images).
            :param step: Step size for selecting images from folders. Default=1 (use all images)
            :param start_index: Start index for selecting images from folders. Default=0
            :param del_folders: Whether to delete the original folders after use
            :param cluster_method: If specified, use clustering_method to select images. Default=None
            :param left_right: If active, expects matching images from stereo cameras. Assumes chosen_paths to lead to right images and expects "left" folder next to "right" folder
            :param convert: Optional choice of ["datumaro", "robert"]. Triggers automatic conversion in DevilsYolo format before dataset creation.
            :param convert_id_blacklist: List of class IDs to ignore during conversion. Only used with convert.
            :param merge_ids: A dict of {x: int, y: int} pairs, where x is the class ID to be merged into class ID y
            :param permute_ids: A list of indices used to permute the class IDs. The index of the list represents the old class ID, the value at that index represents the new class ID. This is applied after merging and blacklisting operations.
            :param nc: Optional number of classes (for dataset.yaml)
            :param box_format: Specifier string for a box format ('xyxy', 'cxcywh', etc.)
            :param class_names: Optional List of strings that specify class names in class_id order (for dataset.yaml)
            :param special_classes: Optional Dict of strings with keys in ['lines', 'ellipses'] and values in class_names (for dataset.yaml)

        Returns:
            Path to final dataset
        """

        if split_percentages is None or type(split_percentages) is not list or len(split_percentages) != 3 or not (
                abs(sum(split_percentages) - 1.0) < 1e-6 or abs(sum(split_percentages) - 100.0) < 1e-6):
            print("WARNING: Using default split_percentages [0.8, 0.1, 0.1]")
            split_percentages = [0.8, 0.1, 0.1]
        if any(not isinstance(percent, (int, float)) for percent in split_percentages):
            raise ValueError("split percentages must be numeric")
        if any(not math.isfinite(percent) for percent in split_percentages):
            raise ValueError("split percentages must be finite")
        total_percentage = sum(split_percentages)
        if abs(total_percentage - 100.0) < 1e-6:
            split_percentages = [p / 100.0 for p in split_percentages]
        elif total_percentage > 0:
            split_percentages = [p / total_percentage for p in split_percentages]

        self._validate_resize(resize)

        # Set up dataset folder structure
        # Always using YOLO-Style dataset structure (with train/val/test top-level folders and images/labels subfolders)
        dataset_dir = self.data_root / dataset_name
        resolved_data_root = self.data_root.resolve()
        resolved_dataset_dir = dataset_dir.resolve()
        if resolved_dataset_dir == resolved_data_root or resolved_data_root not in resolved_dataset_dir.parents:
            raise ValueError(f"Dataset destination must be inside data_root: {dataset_dir}")
        if os.path.exists(dataset_dir):
            raise ValueError(f"Dataset directory {dataset_dir} already exists. Please choose a different name or delete the existing folder.")
        splits = ["train", "val", "test"]
        split_paths = [dataset_dir / split for split in splits]

        if step <= 0:
            raise ValueError("step must be a positive integer")
        if type(start_index) is not int or start_index < 0:
            raise ValueError("start_index must be a non-negative integer")
        if any(percent < 0 for percent in split_percentages) or sum(split_percentages) <= 0:
            raise ValueError("split percentages must be non-negative and have a positive total")
        if abs(sum(split_percentages) - 1.0) > 1e-6:
            raise ValueError("split percentages must sum to 1")
        if convert == "obb":
            raise ValueError(
                "OBB annotations are not supported by MaesyDataset; use 'dataset convert --input-format obb' for standalone Ultralytics export"
            )
        if convert and left_right:
            raise ValueError("left_right is not supported together with dataset conversion")

        converted_roots = []
        conversion_metadata = None
        conversion_context = tempfile.TemporaryDirectory(prefix="maesy-convert-") if convert else None
        staging_dir = None
        try:
            if convert:
                match convert:
                    case "datumaro":
                        from _maesy_core.dataset import datumaro_to_devils_yolo
                        conversion_function = datumaro_to_devils_yolo
                    case "robert":
                        from _maesy_core.dataset import robert_to_devils_yolo
                        conversion_function = robert_to_devils_yolo
                    case _:
                        raise ValueError(f"Unknown dataset type {convert}. Conversion failed...")

                for index, folder in enumerate(folder_names):
                    converted_root = Path(conversion_context.name) / f"source-{index}"
                    _root, converted_nc, converted_names, converted_special = conversion_function(
                        folder,
                        box_format=box_format,
                        class_id_blacklist=convert_id_blacklist,
                        merge_ids=merge_ids,
                        permute_ids=permute_ids,
                        output_path=converted_root,
                    )
                    metadata = (converted_nc, converted_names, converted_special)
                    if conversion_metadata is not None and metadata != conversion_metadata:
                        raise ValueError("All converted input folders must have identical class metadata")
                    conversion_metadata = metadata
                    converted_roots.append(converted_root)

            def _source_label_path(image_path: Path) -> Path:
                image_path = image_path.resolve()
                for folder in folder_names:
                    source_root = Path(folder).resolve()
                    if image_path.is_relative_to(source_root):
                        relative_image = image_path.relative_to(source_root)
                        for images_dir in (source_root, *source_root.parents):
                            if images_dir.name.casefold() == "images":
                                relative_to_images = image_path.relative_to(images_dir)
                                return images_dir.parent / "labels" / relative_to_images.with_suffix(".txt")
                        return image_path.with_suffix(".txt")
                return image_path.with_suffix(".txt")

            def _left_image_path(image_path: Path) -> Path:
                image_path = image_path.resolve()
                for folder in folder_names:
                    right_root = Path(folder).resolve()
                    if image_path.is_relative_to(right_root):
                        relative_image = image_path.relative_to(right_root)
                        left_name = f"{relative_image.stem.removesuffix('_right')}_left{relative_image.suffix}"
                        return right_root.parent / "Left" / relative_image.parent / left_name
                raise ValueError(f"Right image {image_path} is not under an input folder")

            def _get_paths():
                img_paths = []
                if cluster_method is None:
                    for folder in folder_names:
                        folder_path = Path(folder)
                        if not folder_path.exists() or not folder_path.is_dir():
                            print(f"WARNING: Folder {folder} does not exist or is not a directory. Skipping...")
                            continue
                        image_paths = sorted(
                            path.resolve() for path in folder_path.rglob("*")
                            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        )
                        image_paths = image_paths[start_index::step]
                        if not image_paths:
                            print(f"WARNING: No image files found in folder {folder}. Skipping...")
                        img_paths.extend(image_paths)
                    return img_paths

                selected = self.cluster_data(
                    folder_names, chosen_paths, cluster_method=cluster_method, step=step,
                    start_index=start_index, batch_size=cluster_batch_size,
                    num_clusters=num_clusters, similarity_threshold=similarity_threshold,
                )
                image_paths = [
                    Path(path).resolve() for path in selected
                    if Path(path).is_file() and Path(path).suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
                if not image_paths:
                    raise ValueError("No image files found after clustering")
                return image_paths

            source_label_files = None
            if convert:
                converted_pairs = []
                converted_image_map = {}
                converted_image_roots = []
                for converted_root in converted_roots:
                    if convert == "datumaro":
                        candidate_splits = [(
                            converted_root / "default" / "images",
                            converted_root / "default" / "labels",
                        )]
                    else:
                        candidate_splits = [
                            (converted_root / split / "images", converted_root / split / "labels")
                            for split in splits
                            if (converted_root / split / "images").is_dir()
                        ]
                    for image_root, label_root in candidate_splits:
                        converted_image_roots.append(image_root)
                        for image_file in sorted(
                            path for path in image_root.rglob("*")
                            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
                        ):
                            relative = image_file.relative_to(image_root)
                            label_file = label_root / relative.with_suffix(".txt")
                            converted_pairs.append((image_file, label_file))
                            converted_image_map[image_file.resolve()] = label_file
                if cluster_method is not None:
                    selected = self.cluster_data(
                        [str(root) for root in converted_image_roots], chosen_paths,
                        cluster_method=cluster_method, step=step, start_index=start_index,
                        batch_size=cluster_batch_size, num_clusters=num_clusters,
                        similarity_threshold=similarity_threshold,
                    )
                    selected_paths = [Path(path) for path in selected]
                    converted_pairs = [
                        (image, converted_image_map[image.resolve()])
                        for image in selected_paths
                        if image.resolve() in converted_image_map
                    ]
                    if not converted_pairs:
                        raise ValueError("No converted images found after clustering")
                else:
                    # Match the non-converted behavior: apply sampling independently
                    # to each input tree before combining the selected examples.
                    pairs_by_root = []
                    for image_root in converted_image_roots:
                        root_pairs = [
                            pair for pair in converted_pairs
                            if pair[0].is_relative_to(image_root)
                        ]
                        pairs_by_root.extend(root_pairs[start_index::step])
                    converted_pairs = pairs_by_root
                image_files = [image for image, _label in converted_pairs]
                source_label_files = [label for _image, label in converted_pairs]
            else:
                image_files = _get_paths()
                source_label_files = [_source_label_path(image) for image in image_files]
                if cluster_method is None:
                    image_label_pairs = []
                    seen_images = set()
                    folder_paths = sorted(
                        (Path(folder).resolve() for folder in folder_names),
                        key=lambda path: len(path.parts),
                        reverse=True,
                    )
                    for folder_path in folder_paths:
                        folder_pairs = [
                            (image, _source_label_path(image))
                            for image in image_files
                            if image.is_relative_to(folder_path) and image not in seen_images
                        ]
                        image_label_pairs.extend(folder_pairs)
                        seen_images.update(image for image, _label in folder_pairs)
                else:
                    image_label_pairs = list(zip(image_files, source_label_files))
            if not image_files:
                raise ValueError("No image files found for dataset creation")
            if not convert:
                image_files = [image for image, _label in image_label_pairs]
                source_label_files = [label for _image, label in image_label_pairs]

            # Keep each image and its label together while assigning dataset splits.
            if convert:
                image_label_pairs = list(zip(image_files, source_label_files))
            random.shuffle(image_label_pairs)
            num_images = len(image_label_pairs)
            train_count, val_count, test_count = _allocate_split_counts(
                num_images, split_percentages
            )
            assignments = [
                split_path
                for split_path, count in zip(split_paths, (train_count, val_count, test_count))
                for _ in range(count)
            ]

            image_files = [image for image, _label in image_label_pairs]
            source_label_files = [label for _image, label in image_label_pairs]
            image_targets = []
            label_targets = [] if with_labels else None
            for image_file, split_path in zip(image_files, assignments):
                # Flatten nested source IDs; _copy_resize resolves basename collisions
                # by renaming the image and label together.
                if convert:
                    root = next(
                        candidate for candidate in converted_image_roots
                        if image_file.is_relative_to(candidate)
                    )
                    target_relative = image_file.relative_to(root)
                else:
                    target_relative = Path(image_file.name)
                image_targets.append(split_path / "images" / target_relative)
                if label_targets is not None:
                    label_targets.append(split_path / "labels" / target_relative.with_suffix(".txt"))

            left_images = None
            if left_right:
                left_images = [_left_image_path(path) for path in image_files]
                missing_left_images = [path for path in left_images if not path.is_file()]
                if missing_left_images:
                    raise FileNotFoundError(
                        f"Missing {len(missing_left_images)} corresponding left-camera image(s), "
                        f"including {missing_left_images[0]}"
                    )

            staging_dir = Path(tempfile.mkdtemp(
                prefix=f".{Path(dataset_name).name}.staging-", dir=self.data_root
            ))
            staging_split_paths = [staging_dir / split for split in splits]
            staging_image_targets = [
                staging_dir / assignment.name / "images"
                / target.relative_to(assignment / "images")
                for target, assignment in zip(image_targets, assignments)
            ]
            staging_label_targets = [
                staging_dir / assignment.name / "labels"
                / target.relative_to(assignment / "labels")
                for target, assignment in zip(label_targets, assignments)
            ] if label_targets is not None else None
            for split_path in staging_split_paths:
                (split_path / "images").mkdir(parents=True, exist_ok=True)
                if with_labels:
                    (split_path / "labels").mkdir(parents=True, exist_ok=True)
            final_image_targets = self._copy_resize(
                image_files, staging_image_targets, resize, staging_label_targets,
                require_labels=bool(convert and with_labels),
                source_label_paths=source_label_files,
            )
            final_image_targets = [
                image_target if image_target is not None else target
                for image_target, target in zip(final_image_targets, staging_image_targets)
            ]

            if left_right:
                print("Copying corresponding left files")
                left_targets = [
                    target.parent / f"{target.stem.removesuffix('_right')}_left{target.suffix}"
                    for target in final_image_targets
                ]
                left_label_targets = [
                    target.parent.parent / "labels" / f"{target.stem}.txt"
                    for target in left_targets
                ] if label_targets is not None else None
                left_label_sources = [_source_label_path(path) for path in left_images]
                self._copy_resize(
                    left_images, left_targets, resize, left_label_targets,
                    source_label_paths=left_label_sources,
                )
            if conversion_metadata is not None:
                nc, class_names, special_classes = conversion_metadata
            print("Creating dataset.yaml")
            self._create_yaml(
                staging_dir, num_classes=nc, splits=splits, class_names=class_names,
                special_classes=special_classes, box_format=box_format, yaml_root=dataset_dir,
            )
            staging_dir.rename(dataset_dir)
            staging_dir = None
            print("Creation successful!")
            return dataset_dir
        finally:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            if conversion_context is not None:
                conversion_context.cleanup()

    def list_datasets(self) -> list:
        """
        List all datasets in data root.
        
        Returns:
            List of dataset names
        """
        if not self.data_root.exists():
            return []
        return [d.name for d in self.data_root.iterdir() if d.is_dir()]
