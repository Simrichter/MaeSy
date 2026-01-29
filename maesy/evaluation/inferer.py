import torch
from tqdm import tqdm
from ..training.utils import handle_raw_batch


class Inferer:
    """
    Class for running inference on a model over a dataset.
    """

    def __init__(
        self,
        model,
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
    def infer(self):
        """
        Get the models outputs of a complete epoch.

        Returns:
            tensor with the results. First dimension is the batch/epoch dimension.
        """

        if self.data_loader is None:
            raise ValueError("Data loader is required for evaluation. Provide data_loader during initialization.")

        all_predictions = []
        all_targets = []

        print("Running inference...")
        for batch in tqdm(self.data_loader):
            images, targets = handle_raw_batch(batch, self.device)

            # Get predictions
            predictions = self.model.forward(images)

            all_predictions.extend(predictions)
            all_targets.extend(targets)
        print("Done.")
        return all_predictions, all_targets