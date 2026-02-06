import torch
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
        data_loader=None,
        device=torch.device("cuda")
    ):
        """
        Initialize inferer.

        Args:
            model: Model to run inference on
            data_loader: Data loader for inference (optional for single batch inference)
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
            tensor with the results. First dimension is the batch/epoch dimension.
        """

        if self.data_loader is None:
            raise ValueError("Data loader is required for evaluation. Provide data_loader during initialization.")

        all_predictions = []
        all_targets = []

        # TODO: Cuda support not very efficient. Maybe due to list? Nope, probably only on HDD. But need to verify
        print(f"Running inference on {len(self.data_loader)} batches... (Device: {self.device})")
        for batch in tqdm(self.data_loader):
            images, targets = handle_raw_batch(batch, self.device)
            img_preds, targets = self.model.infer(images, targets)

            all_predictions.append(img_preds.detach())
            all_targets.append(targets)
        print("Done.")
        return all_predictions, all_targets