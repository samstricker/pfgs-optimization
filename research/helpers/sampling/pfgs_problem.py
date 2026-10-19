"""This module defines the Himmelblau problem class for PFGS."""

from typing import List, Tuple

from numpy.typing import NDArray
import polars as pl
import numpy as np
from scipy.stats import norm

from paretodo import (
    BaseProblem,
    InvestigationSpace,
    OptimizationAxis,
    OptimizationProblem,
)
from helpers.data.dataset import Dataset
from helpers.botorch_gp import BoTorchGP


class GenericProblem(BaseProblem):
    """ "This class defines the Himmelblau problem for PFGS.

    Parameters
    ----------
    investigation_space : InvestigationSpace
        The investigation space for the problem.
    dataset : Dataset
        The dataset containing the initial data for the problem.
    model : BoTorchGP
        The Gaussian Process model used for predictions.
    optimization_problem : OptimizationProblem
        The optimization problem defining the objectives and constraints.
    """

    def __init__(
        self,
        investigation_space: InvestigationSpace,
        dataset: Dataset,
        model: BoTorchGP,
        optimization_problem: OptimizationProblem,
        robustness_confidence: float,
    ):
        super().__init__(
            investigation_space=investigation_space,
            optimization_problem=optimization_problem,
        )
        self.robustness_confidence = robustness_confidence
        self.dataset = dataset
        self.model = model

    def predict(self, x, normalized: bool = True) -> Tuple[Dataset, pl.DataFrame]:
        """Predict the objective values for a given input x.

        Parameters
        ----------
        x : NDArray[np.float64]
            The input data for which to make predictions. Shape should be (n_samples, n_features).
        normalized : bool, optional
            Whether to return normalized predictions (default is True).

        Returns
        -------
        Tuple[Dataset, pl.DataFrame]
            A tuple containing the Dataset with predictions and a DataFrame of the predictions.
        """
        if x.shape[1] != len(self.dataset.x_labels):
            raise ValueError("Input dimension does not match dataset x_labels length.")
        data = self.dataset.copy()
        data.dataset_type = "recipes"
        data.y_data = None
        data.y_errors = None
        data.normalized = normalized
        x_rounded = self.investigation_space.round(x)
        data.x_data = x_rounded
        predictions = self.model.predict(data)
        return predictions, predictions.to_dataframe()

    def predict_pso(self, x) -> Tuple[Dataset, pl.DataFrame]:
        """Predict the objective values for a given input x without normalization.

        Parameters
        ----------
        x : NDArray[np.float64]
            The input data for which to make predictions. Shape should be (n_samples, n_features).

        Returns
        -------
        Tuple[Dataset, pl.DataFrame]
            A tuple containing the Dataset with predictions and a DataFrame of the predictions.
        """
        if x.shape[1] != len(self.dataset.x_labels):
            raise ValueError("Input dimension does not match dataset x_labels length.")
        data = self.dataset.copy()
        data.dataset_type = "recipes"
        data.y_data = None
        data.y_errors = None
        data.normalized = False
        data.x_data = x
        predictions = self.model.predict(data)
        return predictions, predictions.to_dataframe()

    def prediction_to_robust_objective_values(
        self, predictions, opt_axis: OptimizationAxis
    ) -> NDArray[np.float64]:
        multiplier = norm.ppf(1 - (1 - self.robustness_confidence) / 2)
        if predictions.y_labels is None or predictions.y_data is None:
            raise ValueError("Predictions must contain y_labels and y_data.")
        if predictions.normalized is True:
            predictions.denormalize()
        if opt_axis.var_name not in predictions.y_labels:
            raise ValueError(
                f"Optimization axis {opt_axis.var_name} not found in predictions {predictions.y_labels}."
            )
        idx = predictions.y_labels.index(opt_axis.var_name)
        lcb = predictions.y_data[:, idx] - multiplier * predictions.y_errors[:, idx]
        return lcb

    def create_objective_list(
        self,
        predictions: Dataset,
        for_minimization: bool = True,
    ) -> List:
        """Create a list of objective values from the predictions.

        Parameters
        ----------
        predictions : Dataset
            The dataset containing the predictions from which to create the objective list.
        for_minimization : bool, optional
            Whether to create the objective list for minimization (default is True).
            If False, the objective values will be negated for maximization.

        Returns
        -------
        List
            A list of objective values corresponding to the optimization problem's objectives.
        """
        objective_list = []
        if (
            predictions.x_data is None
            or predictions.y_labels is None
            or predictions.y_data is None
            or predictions.y_errors is None
        ):
            raise ValueError(
                "Predictions must contain x_data, y_labels, y_data, and y_errors."
            )
        if predictions.normalized is True:
            predictions.denormalize()

        for obj in self.optimization_problem.problem:
            if obj.var_name in predictions.y_labels:
                idx = predictions.y_labels.index(obj.var_name)
                objective_value = self._get_objective_value(
                    obj,
                    x_vals=predictions.x_data,
                    mean=predictions.y_data[:, idx],
                    sigma=predictions.y_errors[:, idx],
                    for_minimization=for_minimization,
                )
            else:
                raise ValueError(
                    f"Objective: {obj.var_name} not found in predictions {predictions.y_labels}."
                )
            objective_list.append(objective_value)
        return objective_list

    def _predict_for_pso(
        self,
        objective: OptimizationAxis,
    ):
        """Create a prediction function for PSO optimization based on the given objective."""

        def obj_function(x: NDArray[np.float64]):
            """Calculate the objective value for a given input x."""
            if objective.optimization_type.lower() != "robustness":
                raise ValueError(
                    f"Unsupported optimization type {objective.optimization_type}"
                    f" for objective {objective.var_name}. Only 'robustness' is supported."
                )
            predictions, _ = self.predict_pso(x)
            if predictions.y_labels is None or predictions.y_data is None:
                raise ValueError("Predictions must contain y_labels and y_data.")
            if predictions.normalized is True:
                predictions.denormalize()
            if objective.var_name in predictions.y_labels:
                idx = predictions.y_labels.index(objective.var_name)
                objective_value = predictions.y_data[:, idx]
                if (
                    objective.minimize is True
                ):  # For minimization, we maximize to get the worst case scenario
                    objective_value = -objective_value
                return objective_value
            else:
                raise ValueError(
                    f"Objective variable name {objective.var_name} "
                    f"not found in predictions {predictions.y_labels}."
                )

        return obj_function
