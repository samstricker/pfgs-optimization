"""Tests for the BaseProblem class."""

from typing import Any, List, Tuple

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from paretodo import (
    BaseProblem,
    InvestigationSpace,
    InvestigationSpaceVariable,
    OptimizationProblem,
    OptimizationAxis,
)


class SimpleProblem(BaseProblem):
    """Simple concrete implementation of BaseProblem for testing."""

    def predict(
        self, x: NDArray[np.float64], normalized: bool = True
    ) -> Tuple[Any, pl.DataFrame]:
        """Simple mock prediction."""
        # x is normalized to [-1, 1], convert to real values
        predictions = x * 2  # Simple scaling
        df = pl.DataFrame(
            {
                "pred_col": (
                    predictions[:, 0].tolist()
                    if predictions.ndim > 1
                    else [predictions[0]]
                ),
            }
        )
        return predictions, df

    def create_objective_list(self, predictions, for_minimization: bool = True) -> List:
        """Create objectives from predictions."""
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        objective_list = [predictions[:, 0].copy()]
        if len(self.optimization_problem.problem) > 1:
            objective_list.append(predictions[:, 0] ** 2)

        return objective_list

    def prediction_to_robust_objective_values(
        self, predictions, opt_axis: OptimizationAxis
    ) -> NDArray[np.float64]:
        """Mock robust objective values: first prediction column."""
        return np.asarray(predictions).reshape(len(predictions), -1)[:, 0]


class TestBaseProblem:
    """Tests for BaseProblem class."""

    def test_base_problem_initialization(self):
        """Test BaseProblem initialization."""
        var = InvestigationSpaceVariable(
            name="x",
            min=0.0,
            max=10.0,
            discretization=1.0,
        )
        space = InvestigationSpace(variables=[var])

        axis = OptimizationAxis(
            var_name="x",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis])

        base_problem = SimpleProblem(
            investigation_space=space,
            optimization_problem=problem,
        )

        assert base_problem.investigation_space == space
        assert base_problem.optimization_problem == problem
        assert base_problem.n_var == 1
        assert base_problem.n_obj == 1

    def test_base_problem_multi_variable(self):
        """Test BaseProblem with multiple variables."""
        var1 = InvestigationSpaceVariable(
            name="x",
            min=0.0,
            max=10.0,
            discretization=1.0,
        )
        var2 = InvestigationSpaceVariable(
            name="y",
            min=-5.0,
            max=5.0,
            discretization=0.5,
        )
        space = InvestigationSpace(variables=[var1, var2])

        axis1 = OptimizationAxis(
            var_name="x",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        axis2 = OptimizationAxis(
            var_name="y",
            minimize=False,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis1, axis2])

        base_problem = SimpleProblem(
            investigation_space=space,
            optimization_problem=problem,
        )

        assert base_problem.n_var == 2
        assert base_problem.n_obj == 2

    def test_set_best_values(self):
        """Test setting best values."""
        var = InvestigationSpaceVariable(
            name="x",
            min=0.0,
            max=10.0,
            discretization=1.0,
        )
        space = InvestigationSpace(variables=[var])

        axis = OptimizationAxis(
            var_name="x",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis])

        base_problem = SimpleProblem(
            investigation_space=space,
            optimization_problem=problem,
        )

        best_values = {"x": 5.0}
        base_problem.set_best_values(best_values)

        assert base_problem.best_values == best_values

    def test_set_best_values_invalid_type(self):
        """Test setting best values with invalid type."""
        var = InvestigationSpaceVariable(
            name="x",
            min=0.0,
            max=10.0,
            discretization=1.0,
        )
        space = InvestigationSpace(variables=[var])

        axis = OptimizationAxis(
            var_name="x",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis])

        base_problem = SimpleProblem(
            investigation_space=space,
            optimization_problem=problem,
        )

        with pytest.raises(ValueError, match="best_values must be a dictionary"):
            base_problem.set_best_values([5.0])

    def test_predict_method(self):
        """Test predict method."""
        var = InvestigationSpaceVariable(
            name="x",
            min=0.0,
            max=10.0,
            discretization=1.0,
        )
        space = InvestigationSpace(variables=[var])

        axis = OptimizationAxis(
            var_name="x",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis])

        base_problem = SimpleProblem(
            investigation_space=space,
            optimization_problem=problem,
        )

        x = np.array([[0.5], [-0.5]])
        _predictions, df = base_problem.predict(x)

        assert df.shape[0] == x.shape[0]
        assert isinstance(df, pl.DataFrame)
        assert len(df) == len(x)

    def test_create_objective_list(self):
        """Test create_objective_list method."""
        var = InvestigationSpaceVariable(
            name="x",
            min=0.0,
            max=10.0,
            discretization=1.0,
        )
        space = InvestigationSpace(variables=[var])

        axis = OptimizationAxis(
            var_name="x",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis])

        base_problem = SimpleProblem(
            investigation_space=space,
            optimization_problem=problem,
        )

        predictions = np.array([[0.5], [1.5]])
        objectives = base_problem.create_objective_list(predictions)

        assert isinstance(objectives, list)
        assert len(objectives) == 1
