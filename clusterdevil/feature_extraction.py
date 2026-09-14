from typing import Optional, List
from _maesy_core.model import BaseModel
from clusterdevil.utils import store_features, find_and_load_features


def extract_features(model: BaseModel, paths: List[str], device:Optional[str]=None, batch_size=1):
    """
    Extract features from a model given a dataloader and device.
    :param model: The model to extract features from
    :param paths: List of paths to the datasets to extract features from
    :param device: The device to run the model on (e.g., 'cuda' or 'cpu')
    :param batch_size: The batch size for the dataloader
    :return: A list of extracted features
    """
    from _maesy_core.dataset import MaesyDataset, MultiDataset
    from _maesy_core.inference.inferer import Inferer
    from torch.utils.data import DataLoader
    from _maesy_core.dataset.augmentations import ClusterTransforms
    import torch

    # Check if features already exist for the given paths and model hash
    model_hash = model.get_model_hash()
    final_features, loaded_paths = find_and_load_features(paths, model_hash)
    if len(loaded_paths) > 0:
        print(f"Found existing features for model hash '{model_hash}' in the following paths:")
        for loaded_path in loaded_paths:
            print(f" - {loaded_path}")
    else:
        print(f"No existing features found for model hash '{model_hash}' in the provided paths.")
    paths = [path for path in paths if not any(path.startswith(loaded_path.removesuffix(f"features_{model_hash}.feat")) for loaded_path in loaded_paths)]

    model.eval()
    # assert in_dims[-2] == in_dims[-1], "Failed, only models with square input dimensions are supported (height == width)"

    if len(paths) > 0:
        in_dims = model.get_input_dims()
        img_transforms = ClusterTransforms(image_height=in_dims[-2], image_width=in_dims[-1])
        # Create dataset from all image directories
        print(f"Extracting features for {len(paths)} paths...")
        internal_dataset = MultiDataset([MaesyDataset(dataset_dir=path, annotation_type="image_folder", transforms=img_transforms, use_first_n=None) for path in paths])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

        new_dataloader = DataLoader(internal_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory="cuda" in str(device), drop_last=False, in_order=True)

        inferer = Inferer(model, new_dataloader, device=device)
        preds, _ = inferer.infer()

        paths = [str(internal_dataset.get_image_path(i)) for i in range(len(internal_dataset))]
        final_features.update({path: feature for path, feature in zip(paths, torch.cat([p["c6"] for p in preds], dim=0))})
    else:
        print("No further paths to extract features from.")
    store_features(final_features, model_hash)
    return final_features