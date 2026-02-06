from torch.utils.data import Dataset


class MultiDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        """
        A dataset that combines multiple datasets.

        :param datasets: List of datasets to combine
        """
        self.datasets = datasets
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