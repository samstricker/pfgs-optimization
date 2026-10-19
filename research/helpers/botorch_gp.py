"""This module defines the SKlearnGP model class for Gaussian Process regression using scikit-learn."""

from typing import List, Optional

import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from helpers.data.dataset import Dataset


class BoTorchGP:
    """Gaussian Process regression model using BoTorch."""

    def __init__(self):
        self.models: List[SingleTaskGP] = []
        self.train_dataset: Optional[Dataset] = None
        self.mll = None
        self.optimizer = None
        torch.manual_seed(11)

    def train(self, dataset: Dataset) -> None:
        """Train the Gaussian Process model using the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing the training data. The dataset will be normalized before training.
        """
        dataset.normalize()
        self.train_dataset = dataset.copy()
        x_tensor = torch.tensor(dataset.x_data, dtype=torch.float64)
        y_data = torch.tensor(dataset.y_data, dtype=torch.float64)
        num_tasks = y_data.shape[1] if y_data.ndim > 1 else 1
        self.models = []
        for task_idx in range(num_tasks):
            y_tensor = y_data[:, task_idx : task_idx + 1]
            model = SingleTaskGP(
                train_X=x_tensor,
                train_Y=y_tensor,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            self.models.append(model)

    def predict(self, dataset: Dataset) -> Dataset:
        """Predict using the trained Gaussian Process model.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing the input data for prediction.
            The dataset will be normalized before prediction.

        Returns
        -------
        Dataset
            A new dataset containing the predictions and associated uncertainties.
            The predictions will be denormalized before being returned.
        """
        dataset.normalize()
        x_tensor = torch.tensor(dataset.x_data, dtype=torch.float64)

        with torch.no_grad():
            predictions_list = []
            std_list = []
            for model in self.models:
                model.eval()
                posterior = model.posterior(x_tensor)
                y_mean = posterior.mean.detach().cpu().numpy()
                y_var = posterior.variance.detach().cpu().numpy()
                predictions_list.append(y_mean)
                std_list.append(y_var**0.5)

        # Stack predictions and standard deviations
        y_mean = np.column_stack(predictions_list)
        y_std = np.column_stack(std_list)
        predictions = dataset.copy()
        predictions.dataset_type = "predictions"
        predictions.y_data = y_mean
        predictions.y_errors = y_std
        predictions.denormalize()
        return predictions
