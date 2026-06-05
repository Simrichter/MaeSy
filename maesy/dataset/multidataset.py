from typing import Dict

from torch.utils.data import Dataset

from maesy.dataset import MaesyDataset


class MultiDataset(Dataset):
    def __init__(self, datasets: list[MaesyDataset]):
        """
        A dataset that combines multiple datasets.

        :param datasets: List of datasets to combine
        """
        self.datasets = datasets
        special_classes = [d.get_special_classes() for d in datasets]
        assert special_classes.count(special_classes[0]) == len(datasets), "Failed to instantiate multi-dataset. Not all datasets in list are compatible regarding special_classes"
        nc = [d.get_num_classes() for d in datasets]
        assert nc.count(nc[0]) == len(datasets), "Failed to instantiate multi-dataset. Not all datasets in list are compatible regarding num_classes"

        self.cumulative_sizes = self.cumsum(self.datasets)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = 0
        while idx >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def get_image_path(self, idx):
        dataset_idx = 0
        while idx >= self.cumulative_sizes[dataset_idx]:
            dataset_idx += 1
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_image_path(sample_idx)

    def get_special_classes(self) -> Dict[str, int]:
        """
            Get all classes that are not standard axis-aligned bounding boxes.
            (This is used for multi-head training)
            Uses get_special_classes() of the first dataset in the list, since init ensures that it is the same for all datasets
            Returns:
                A dict containing {name: class_id} pairs
        """
        return self.datasets[0].get_special_classes()

    def get_num_classes(self) -> int:
        """
        Get the number of classes according to dataset.yaml
        Uses get_num_classes() of the first dataset in the list, since init ensures that it is the same for all datasets
        Returns:
            Number of classes
        """
        return self.datasets[0].get_num_classes()
