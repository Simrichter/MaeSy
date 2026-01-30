"""Dataset manager for downloading and managing datasets."""

import os
import json
import shutil
import zipfile
import requests
from pathlib import Path
from typing import Optional, Dict, Any
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

    def cluster_data(self, folder_names: list[str], num_clusters: int, cluster_method: str = "resnet+kmeans"):
        """
            Clusters all images in the given folders into num_clusters clusters using the specified method.
            The paths of the images to be used are returned as a list

        Arguments:
            :param folder_names: A list of paths to image data folders
            :param num_clusters: The number of clusters to create
            :param cluster_method: A string specifying the clustering method to use. Default is resnet+kmeans
            :return: A list of paths to the selected images
        """
        from maesy.dataset.clustering_methods.resnet_kmeans import cluster
        return cluster(folder_names, n_C=num_clusters)


    def create_dataset(self,
                       folder_names: list[str],
                       dataset_name: str,
                       split_percentages: list[float] = None,
                       resize: list[int] = None,
                       step: int = 1,
                       start_index: int = 0,
                       del_folders: bool = False,
                       cluster_method = None
                       ) -> Path:
        """
        Combines multiple folders with images into a single dataset

        Arguments:
            :param folder_names: List of folders that contain images
            :param dataset_name: Name of the dataset
            :param split_percentages: List of percentages for the data subsets in format [train, val, test]. Defaults to [0.8, 0.1, 0.1] if not/incorrectly specified
            :param resize: Resize images to WIDTH HEIGHT or WIDTH² if HEIGHT not specified
            :param step: Step size for selecting images from folders. Default=1 (use all images)
            :param start_index: Start index for selecting images from folders. Default=0
            :param del_folders: Whether to delete the original folders after use
            :param cluster_method: If specified, use clustering_method to select images. Default=None

        Returns:
            Path to final dataset
        """

        if split_percentages is None or type(split_percentages) is not list or len(split_percentages)!=3 or not abs(sum(split_percentages) - 1.0) < 1e-6:
            print("WARNING: Using default split_percentages [0.8, 0.1, 0.1]")
            split_percentages = [0.8, 0.1, 0.1]

        # path = self.download_data(url, dataset_name, extract, force)

        dataset_dir = self.data_root / dataset_name
        train_path = dataset_dir / "train"
        val_path = dataset_dir / "val"
        test_path = dataset_dir / "test"
        os.makedirs(dataset_dir, exist_ok=True)
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(val_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        if cluster_method is None:
            # TODO: Make nice (loop over folders and call single method "handle_folder" or so)
            for folder in folder_names:
                folder_path = Path(folder)
                if not folder_path.exists() or not folder_path.is_dir():
                    print(f"WARNING: Folder {folder} does not exist or is not a directory. Skipping...")
                    continue

                image_files = [f for f in folder_path.iterdir() if
                               f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                num_images = len(image_files)

                if num_images == 0:
                    print(f"WARNING: No image files found in folder {folder}. Skipping...")
                    continue

                # Shuffle images
                random.shuffle(image_files)

                # Calculate split indices
                train_end = int(split_percentages[0] * num_images)
                val_end = train_end + int(split_percentages[1] * num_images)

                # Copy files to respective folders
                for i, img_file in enumerate(tqdm(image_files, desc="Copying files"), start=start_index):
                    if i%step != 0: #TODO: Also make this option possible with clustering?
                        continue
                    if i < train_end:
                        dest_path = train_path / img_file.name
                    elif i < val_end:
                        dest_path = val_path / img_file.name
                    else:
                        dest_path = test_path / img_file.name

                    if resize is None:
                        # os.rename(img_file, dest_path)
                        shutil.copy(img_file, dest_path)
                    else:
                        from PIL import Image
                        with Image.open(img_file) as img:
                            if len(resize) == 2:
                                img = img.resize((resize[0], resize[1]))
                            else:
                                print("WARNING: Resize parameter must be a list of two integers. Skipping resizing.")
                                shutil.copy(img_file, dest_path)
                                continue
                            img.save(dest_path)

                if del_folders:
                    print("Deleting original folder:", folder_path)
                    shutil.rmtree(folder_path)
                    # os.rmdir(folder_path)
            # TODO: Add support for label files
            return dataset_dir
        else:
            folder_path = self.cluster_data(folder_names, cluster_method=cluster_method, num_clusters=100)

            image_files = [f for f in folder_path.iterdir() if
                           f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            num_images = len(image_files)

            if num_images == 0:
                print(f"WARNING: No image files found after clustering. Exiting...")
                return dataset_dir

            # Shuffle images
            random.shuffle(image_files)

            # Calculate split indices
            train_end = int(split_percentages[0] * num_images)
            val_end = train_end + int(split_percentages[1] * num_images)

            # Copy files to respective folders
            for i, img_file in enumerate(tqdm(image_files, desc="Copying files")):
                if i < train_end:
                    dest_path = train_path / img_file.name
                elif i < val_end:
                    dest_path = val_path / img_file.name
                else:
                    dest_path = test_path / img_file.name

                if resize is None:
                    shutil.copy(img_file, dest_path)
                else:
                    from PIL import Image
                    with Image.open(img_file) as img:
                        if len(resize) == 2:
                            img = img.resize((resize[0], resize[1]))
                        else:
                            print("WARNING: Resize parameter must be a list of two integers. Skipping resizing.")
                            shutil.copy(img_file, dest_path)
                            continue
                        img.save(dest_path)

            if del_folders:
                print("Deleting original folder:", folder_path)
                shutil.rmtree(folder_path)
        # TODO: Add support for label files
        return dataset_dir

    def load_coco_annotations(self, annotation_file: str) -> Dict[str, Any]:
        """
        Load COCO format annotations.
        
        Args:
            annotation_file: Path to COCO annotation JSON file
            
        Returns:
            Dictionary containing COCO annotations
        """
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        return annotations

    def prepare_dataset(
            self,
            dataset_name: str,
            images_dir: str,
            annotations_file: str,
            split: str = "train"
    ) -> Dict[str, Any]:
        """
        Prepare dataset for training/evaluation.
        
        Args:
            dataset_name: Name of the dataset
            images_dir: Directory containing images
            annotations_file: Path to annotations file
            split: Dataset split (train/val/test)
            
        Returns:
            Dictionary with dataset information
        """
        dataset_info = {
            "name": dataset_name,
            "split": split,
            "images_dir": images_dir,
            "annotations_file": annotations_file,
            "num_images": 0,
            "num_annotations": 0,
            "categories": []
        }

        if os.path.exists(annotations_file):
            annotations = self.load_coco_annotations(annotations_file)
            dataset_info["num_images"] = len(annotations.get("images", []))
            dataset_info["num_annotations"] = len(annotations.get("annotations", []))
            dataset_info["categories"] = annotations.get("categories", [])

        return dataset_info

    def list_datasets(self) -> list:
        """
        List all datasets in data root.
        
        Returns:
            List of dataset names
        """
        if not self.data_root.exists():
            return []
        return [d.name for d in self.data_root.iterdir() if d.is_dir()]
