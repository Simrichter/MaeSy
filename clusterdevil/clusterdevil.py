from typing import List



def cluster(model_info: str, dataset_paths: List[str]):
    from _maesy_core.dataset import MaesyDataset, MultiDataset
    from _maesy_core.model.model_tools.model_factory import create_model
    from _maesy_core.inference.inferer import Inferer

    from torch.utils.data import DataLoader
    """
    Cluster images in a dataset using a specified model.

    Args:
        model_info (str): The path to the model to use for feature extraction.
        dataset_paths (List[str]): The paths to the datasets to cluster.
    """
    model = create_model(model_info)
    dataset = MultiDataset([MaesyDataset(dataset_path, "train", "image_folder") for dataset_path in dataset_paths])
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    inferer = Inferer(model, dataset)
    preds, _ = inferer.infer()

    print(preds)