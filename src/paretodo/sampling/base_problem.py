"""Basic optimization problem."""

from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
import polars as pl
from pymoo.core.problem import Problem
import pyswarms as ps


from paretodo.data_models import (
    InvestigationSpace,
    OptimizationAxis,
    OptimizationProblem,
)


class BaseProblem(Problem):
    """Base class for optimization problems.

    Parameters
    ----------
    config : BaseProblemConfig
        Configuration for the problem.
    """

    def __init__(
        self,
        investigation_space: InvestigationSpace,
        optimization_problem: OptimizationProblem,
    ):
        self.investigation_space = investigation_space
        self.optimization_problem = optimization_problem
        super().__init__(
            n_var=investigation_space.n_var,
            n_obj=optimization_problem.n_obj,
            n_constr=0,
            xl=[-1.0] * investigation_space.n_var,
            xu=[1.0] * investigation_space.n_var,
            elementwise=False,
        )
        self.best_values: Dict[str, float] = {}

    def set_best_values(self, best_values: Dict[str, float]) -> None:
        """Set the best values for the optimization axes."""
        if not isinstance(best_values, Dict):

            raise ValueError("best_values must be a dictionary.")
        self.best_values = best_values

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the objective function. (what to minimize)"""
        if np.isnan(x).any():
            print("NaN values found in input x.")
            # x = self._replace_nan_values_by_max(x)
        # x_int = np.round(x)
        # x_real = self.investigation_space.recover_real_values(x_int)
        predictions, _ = self.predict(x)
        objective_list = self.create_objective_list(predictions)
        out["X"] = x
        out["F"] = objective_list

    @abstractmethod
    def predict(self, x: NDArray[np.float64]) -> Tuple[Any, pl.DataFrame]:
        """Predict outcomes for given decision variables."""
        raise NotImplementedError("predict method needs to be implemented by child.")

    @abstractmethod
    def create_objective_list(self, predictions, for_minimization: bool = True) -> List:
        """Create a list of objectives from predictions."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    @abstractmethod
    def _predict_for_pso(self, objective: OptimizationAxis) -> Callable:
        """Predict the objective value for PSO optimization."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_robust_objective_value(
        self, generation: NDArray[np.float64], objective: OptimizationAxis
    ) -> NDArray[np.float64]:
        """Calculate robust objective values accounting for variable uncertainties.

        Parameters
        ----------
        generation : NDArray[np.float64]
            Array of decision variable points.
        objective : OptimizationAxis
            The optimization objective to evaluate.

        Returns
        -------
        NDArray[np.float64]
            Robust objective values for the generation.
        """
        expected_errors = self.investigation_space.expected_errors
        if any(error is None for error in expected_errors):
            raise ValueError(
                "Expected errors must be set for all variables to calculate robust objective value."
            )
        values = []
        for point in generation:
            lower_bounds = point - np.array(expected_errors)
            upper_bounds = point + np.array(expected_errors)
            bounds = (lower_bounds, upper_bounds)
            options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
            n_particles = 20
            n_iters = 100

            optimizer = ps.single.GlobalBestPSO(
                n_particles=n_particles,
                dimensions=len(self.investigation_space.variables),
                options=options,
                bounds=bounds,
            )
            pso_function = self._predict_for_pso(objective)
            best_val, _ = optimizer.optimize(
                objective_func=pso_function,
                iters=n_iters,
                verbose=False,
            )
            values.append(best_val)
        return np.array(values)

    def _get_objective_value(
        self,
        optim_axis: OptimizationAxis,
        x_vals: NDArray[np.float64],
        mean: NDArray[np.float64],
        sigma: NDArray[np.float64],
        for_minimization: bool = True,
    ) -> List:
        if optim_axis.optimization_type == "objective":
            obj = mean
        elif optim_axis.optimization_type == "robustness":
            obj = self.get_robust_objective_value(
                generation=x_vals, objective=optim_axis
            )
        elif optim_axis.optimization_type == "expected_improvement":
            obj = self._calc_expected_improvement(optim_axis, mean, sigma)
        elif optim_axis.optimization_type == "standard_deviation":
            obj = sigma
        elif optim_axis.optimization_type == "constraint":
            obj = self._calculate_probability_of_meeting_constraint(
                mean, sigma, optim_axis.lower_bound, optim_axis.upper_bound
            )
        else:
            raise ValueError("Unknown optimization type.")
        if for_minimization:
            if optim_axis.minimize is False:
                obj = -obj
        return obj.tolist()

    def _calculate_probability_of_meeting_constraint(
        self,
        mean: NDArray[np.float64],
        std: NDArray[np.float64],
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ) -> NDArray[np.float64]:
        """Calculate the probability of meeting a constraint."""

        if lower_bound is not None and upper_bound is not None:
            probability = norm.cdf(upper_bound, mean, std) - norm.cdf(
                lower_bound, mean, std
            )
        elif lower_bound is None and upper_bound is not None:
            probability = norm.cdf(upper_bound, mean, std)
        elif lower_bound is not None and upper_bound is None:
            probability = 1 - norm.cdf(lower_bound, mean, std)
        else:
            raise ValueError("At least one of lower_bound or upper_bound must be set.")
        if isinstance(probability, float):
            probability = np.array([probability])
        elif not isinstance(probability, np.ndarray):
            raise TypeError(
                f"Probability must be a numpy array, got {type(probability)} instead."
            )
        return probability

    def _calc_expected_improvement(
        self,
        optim_axis: OptimizationAxis,
        mean: NDArray[np.float64],
        std: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the expected improvement."""
        if optim_axis.var_name not in self.best_values:
            raise ValueError(
                "Best values must be set before calculating expected improvement."
            )
        best_so_far = self.best_values[optim_axis.var_name]
        probability_improvement = self._calc_probability_of_improvement(
            optim_axis, mean, std
        )
        if optim_axis.minimize is False:
            expected_improvement = (mean - best_so_far) * probability_improvement + (
                std * norm.pdf((mean - best_so_far) / std)
            )
        else:
            expected_improvement = (best_so_far - mean) * probability_improvement + (
                std * norm.pdf((best_so_far - mean) / std)
            )
        return expected_improvement

    def _calc_probability_of_improvement(
        self,
        optim_axis: OptimizationAxis,
        mean: NDArray[np.float64],
        std: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate the probability of improvement."""
        if optim_axis.var_name not in self.best_values:
            raise ValueError(
                "Best values must be set before calculating probability of improvement."
            )
        best_so_far = self.best_values[optim_axis.var_name]
        if optim_axis.minimize is False:
            prob = norm.cdf((mean - best_so_far) / std)
        else:
            prob = norm.cdf((best_so_far - mean) / std)
        return prob

    def _replace_nan_values_by_max(self, data: NDArray):
        """Replace NaN values in data with the maximum integer bounds."""
        max_values = self.investigation_space.int_bounds[1]
        for col in range(data.shape[1]):
            nan_indices = np.isnan(data[:, col])
            data[nan_indices, col] = max_values[col]
        return data
