"""Dataset manager for downloading and managing datasets."""

import os
import json
import zipfile
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import tarfile


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
        
    def download_dataset(
        self,
        url: str,
        dataset_name: str,
        extract: bool = True,
        force: bool = False
    ) -> Path:
        """
        Download dataset from URL.
        
        Args:
            url: URL to download dataset from
            dataset_name: Name of the dataset
            extract: Whether to extract if zip file
            force: Force re-download even if exists
            
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
        if extract and filepath.suffix == '.zip':
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            os.remove(filepath)
        elif extract and filepath.suffix == '.tar':
            print(f"Extracting {filename}...")
            tar = tarfile.open(filepath)
            tar.extractall(dataset_dir)
            tar.close()

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
