"""This module contains the simulation function for the Himmelblau problem."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

from helpers.data.dataset import Dataset


class HimmelblauButterotSimulator:

    def run_simulation(
        self, recipes: Dataset, iteration: Optional[int] = None
    ) -> Dataset:
        """Run the simulation for the given recipes dataset.

        Parameters
        ----------
        recipes : Dataset
            The Dataset containing the recipes to simulate.
        iteration : int
            The current iteration number.

        Returns
        -------
        Dataset
            The Dataset containing the results from the simulation.
        """
        _ = iteration  # Currently not used, but included for consistency with other simulators.
        if recipes.dataset_type != "recipes":
            raise ValueError("Input dataset must be of type 'recipes'.")
        recipes.denormalize()
        experiment_dataset = recipes.copy()
        if recipes.x_data is None:
            raise ValueError("The recipes dataset has no x_data.")
        recipes.normalize()
        hb_data = self._himmelblau_function(recipes.x_data[:, 0], recipes.x_data[:, 1])
        bt_data = self._butterot_function(recipes.x_data[:, 0], recipes.x_data[:, 1])
        experiment_dataset.y_data = np.hstack((hb_data, bt_data))
        experiment_dataset.dataset_type = "experiments"
        experiment_dataset.y_labels = [
            "product_concentration",
            "side_product_concentration",
        ]

        return experiment_dataset

    def _himmelblau_function(self, x1: NDArray, x2: NDArray) -> NDArray:
        """Compute the Himmelblau function for given inputs. Inputs are expected to be [0,1] scaled.

        Parameters
        ----------
        x1 : NDArray
            The first input array.
        x2 : NDArray
            The second input array.

        Returns
        -------
        NDArray
            The computed Himmelblau function values.
        """
        if x1.ndim != 1 or x2.ndim != 1:
            raise ValueError("Input arrays must be one-dimensional.")
        if x1.shape != x2.shape:
            raise ValueError("Input arrays must have the same length.")
        # Map [0,1] inputs to the [-5.5,5.5] domain used for the synthetic landscape.
        x1 = x1 * 11.0 - 5.5
        x2 = x2 * 11.0 - 5.5
        denorm = np.exp(
            8 - 0.01 * ((x1**2 + x2 - 11) ** 2 + (x1 + x2**2 - 7) ** 2 + x1 - 2 * x2)
        )
        return np.array([denorm]).T

    def _butterot_function(self, x1: NDArray, x2: NDArray) -> NDArray:
        """Compute the Butterot function for given inputs. Inputs are expected to be [0,1] scaled.

        Parameters
        ----------
        x1 : NDArray
            The first input array.
        x2 : NDArray
            The second input array.

        Returns
        -------
        NDArray
            The computed Butterot function values.
        """
        if x1.ndim != 1 or x2.ndim != 1:
            raise ValueError("Input arrays must be one-dimensional.")
        if x1.shape != x2.shape:
            raise ValueError("Input arrays must have the same length.")
        # Map [0,1] inputs to the [-5.5,5.5] domain used for the synthetic landscape.
        x1 = x1 * 11.0 - 5.5
        x2 = x2 * 11.0 - 5.5
        denorm = np.sin(x1 + x2) + 0.4 * (x1 - x2) ** 2 - 1.5 * x1 + 2.5 * x2 + 9.0
        denorm = ((denorm / 79.4) + 1.0) * 1720
        return np.array([denorm]).T
