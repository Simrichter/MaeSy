from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.components import Utils
from ..training.utils import handle_raw_batch
from maesy.model import BaseModel


class Inferer:
    """
    Class for running inference on a model over a dataset.
    """

    def __init__(
        self,
        model: BaseModel,
        data_loader: Optional[DataLoader]=None,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ):
        """
        Initialize inferer.

        Args:
            model: Model to run inference on
            data_loader: Data loader for inference
            device: Device to run inference on
        """
        self.model = model
        self.model.eval()
        self.data_loader = data_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def infer(self, **kwargs):
        """
        Get the models outputs of a complete epoch.

        Returns:
            Tuple of Lists. First output are the predictions, second are the corresponding targets.
        """

        if self.data_loader is None:
            raise ValueError("Data loader is required for evaluation. Provide data_loader during initialization.")

        all_predictions = []
        all_targets = []

        print(f"Running inference on {len(self.data_loader)} batches ({len(self.data_loader.dataset)} images)... (Device: {self.device.type})")
        for batch in tqdm(self.data_loader):
            images, targets = handle_raw_batch(batch, self.device)
            _, img_preds, targets = self.model.infer(images, targets, **kwargs)
            all_predictions.extend(img_preds)
            all_targets.extend(targets)
        print("Done.")
        return all_predictions, all_targets