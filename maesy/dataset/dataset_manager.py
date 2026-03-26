"""Dataset manager for downloading and managing datasets."""

import os
import json
import shutil
import zipfile
from math import ceil

import requests
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
import tarfile
import random


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

    def cluster_data(self, folder_names: list[str], chosen_paths: list[str], num_clusters: int,
                     cluster_method: str = "resnet_kmeans", step: int = 1, start_index: int = 0) -> List[Path]:
        """
            Clusters all images in the given folders into num_clusters clusters using the specified method.
            The paths of the images to be used are returned as a list

        Arguments:
            :param folder_names: A list of paths to image data folders
            :param chosen_paths: List of already chosen image paths to include as initial representatives (pass None if not wanted)
            :param num_clusters: The number of clusters to create (for resnet_kmeans)
                                or a proxy for similarity threshold (for sequential_similarity)
            :param cluster_method: A string specifying the clustering method to use. Default is resnet_kmeans
                                   Options: "resnet_kmeans", "sequential_similarity"
            :param step: Step size for selecting images from folders. Default=1 (use all images)
            :param start_index: Start index for selecting images from folders. Default=0
            :return: A list of paths to the selected images
        """
        match cluster_method:
            case "resnet_faiss":
                from maesy.dataset.clustering_methods.resnet_FAISS import cluster_with_faiss as cluster
                # TODO: Make this a parameter?
                similarity_threshold = 0.85
                return cluster(folder_names, chosen_paths, similarity_threshold, step=step, start_index=start_index)
            case "resnet_kmeans":
                from maesy.dataset.clustering_methods.resnet_kmeans import cluster
                return cluster(folder_names, n_c=num_clusters, step=step, start_index=start_index)
            case _:
                raise ValueError(f"Unknown clustering method {cluster_method}")

    @staticmethod
    def _copy_resize(source_paths: List[Path], target_paths: List[Path], resize: int | List[int] | None,
                     label_target_paths: Optional[List[Path]] = None):
        if resize is not None:
            if len(resize) > 2:
                raise ValueError(
                    "Too many values in resize. Resize parameter must either be None or a list of one or two integers: [WIDTH] or [WIDTH HEIGHT]")
            from PIL import Image
            for img_file, target_path in tqdm(zip(source_paths, target_paths),
                                              desc=f"Copying and resizing {len(source_paths)} files"):
                if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    with Image.open(img_file) as img:
                        img = img.resize((resize[0], resize[1] if len(resize) == 2 else resize[0]))
                        if not os.path.exists(target_path.parent):
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                        img.save(target_path)#/ img_file.name
        else:
            for img_file, target_path in tqdm(zip(source_paths, target_paths),
                                              desc=f"Copying {len(source_paths)} files"):
                if not os.path.exists(target_path.parent):
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_file, target_path)

        if label_target_paths is not None:
            for img_file, target_path in tqdm(zip(source_paths, label_target_paths), desc=f"Copying label files"):
                label_file = img_file.with_suffix('.txt')  # Assuming label files have the same name but .json extension
                if label_file.exists():
                    shutil.copy(label_file, target_path) #/ label_file.name

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
                       cluster_method
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

        Returns:
            Path to final dataset
        """

        if split_percentages is None or type(split_percentages) is not list or len(split_percentages) != 3 or not (
                abs(sum(split_percentages) - 1.0) < 1e-6 or abs(sum(split_percentages) - 100.0) < 1e-6):
            print("WARNING: Using default split_percentages [0.8, 0.1, 0.1]")
            split_percentages = [0.8, 0.1, 0.1]
        if abs(sum(split_percentages) - 100.0) < 1e-6:
            split_percentages = [p / 100.0 for p in split_percentages]

        # Set up dataset folder structure
        # Always assuming YOLO-Style dataset structure (with train/val/test top-level folders and images/labels subfolders)
        # only creating the split folders that are actually needed based on split_percentages
        dataset_dir = self.data_root / dataset_name
        split_paths = [dataset_dir / split for i, split in enumerate(["train", "val", "test"]) if
                       split_percentages[i] > 0]
        os.makedirs(dataset_dir, exist_ok=True)
        for split_path in split_paths:
            os.makedirs(split_path / "images", exist_ok=True)
            if with_labels:
                os.makedirs(split_path / "labels", exist_ok=True)

        def get_paths():  # folder_names, cluster_method, start_index=0, step=1
            img_paths = []
            if cluster_method is None:
                for folder in folder_names:
                    folder_path = Path(folder)
                    if not folder_path.exists() or not folder_path.is_dir():
                        print(f"WARNING: Folder {folder} does not exist or is not a directory. Skipping...")
                        continue

                    image_paths = [folder_path / f for f in folder_path.iterdir() if
                                   f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']][start_index::step]
                    if len(image_paths) == 0:
                        print(f"WARNING: No image files found in folder {folder}. Skipping...")
                        continue
                    img_paths.extend(image_paths)
                return img_paths
            else:
                folder_path = self.cluster_data(folder_names, chosen_paths, cluster_method=cluster_method,
                                                num_clusters=500,
                                                step=step, start_index=start_index)
                image_paths = [f for f in folder_path if
                               f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                if len(image_paths) == 0:
                    raise ValueError(f"WARNING: No image files found after clustering. Exiting...")
                return image_paths

        image_files = get_paths()
        num_images = len(image_files)

        # Calculate split indices
        train_end = int(ceil(split_percentages[0] * num_images))
        val_end = train_end + int(ceil(split_percentages[1] * num_images))
        random.shuffle(image_files)

        target_paths = [(split_paths[0] if i < train_end else split_paths[1] if i < val_end else split_paths[
            2]) / f"images/Right/{img_file.name}" for i, img_file in enumerate(image_files)]
        if with_labels:
            target_paths_lbls = [(split_paths[0] if i < train_end else split_paths[1] if i < val_end else split_paths[
                2]) / f"labels/{img_file.with_suffix('.txt').name}" for i, img_file in enumerate(image_files)]
        else:
            target_paths_lbls = None

        self._copy_resize(image_files, target_paths, resize, target_paths_lbls)

        left_right = True # TODO
        if left_right:
            print("Copying corresponding left files")
            image_files = [f.parent.parent/"Left"/f"{Path(f).name.removesuffix('_right.png')}_left.png" for f in image_files]
            target_paths = [(split_paths[0] if i < train_end else split_paths[1] if i < val_end else split_paths[
            2]) / f"images/Left/{img_file.name}" for i, img_file in enumerate(image_files)]
            self._copy_resize(image_files, target_paths, resize, target_paths_lbls)


        # if del_folders:
        #     print("Deleting original folder:", folder_path)
        #     shutil.rmtree(folder_path)
        # os.rmdir(folder_path) TODO
        return dataset_dir

    def list_datasets(self) -> list:
        """
        List all datasets in data root.
        
        Returns:
            List of dataset names
        """
        if not self.data_root.exists():
            return []
        return [d.name for d in self.data_root.iterdir() if d.is_dir()]
