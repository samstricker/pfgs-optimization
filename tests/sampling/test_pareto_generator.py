"""Tests for ParetoFrontGenerator class."""

from unittest.mock import patch, MagicMock

import numpy as np
import polars as pl

from paretodo import (
    ParetoFrontGenerator,
    InvestigationSpace,
    InvestigationSpaceVariable,
    OptimizationProblem,
    OptimizationAxis,
)

from .test_problem import SimpleProblem


class TestParetoFrontGenerator:
    """Tests for ParetoFrontGenerator class."""

    def test_pareto_front_generator_initialization(self):
        """Test ParetoFrontGenerator initialization."""
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

        generator = ParetoFrontGenerator(
            problem=base_problem,
            seed=42,
            pop_size_per_dim=5,
            n_max_gen=2,
        )

        assert generator.problem == base_problem
        assert generator.seed == 42
        assert generator.pop_size_per_dim == 5
        assert generator.n_max_gen == 2

    def test_pareto_front_generator_default_seed(self):
        """Test ParetoFrontGenerator with default seed."""
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

        generator = ParetoFrontGenerator(problem=base_problem)

        assert generator.seed == 11
        assert generator.pop_size_per_dim == 10
        assert generator.n_max_gen == 50

    @patch("paretodo.sampling.pareto_front_generator.minimize")
    def test_generate_pareto_front(self, mock_minimize):
        """Test generate_pareto_front method."""
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

        generator = ParetoFrontGenerator(
            problem=base_problem,
            seed=42,
            pop_size_per_dim=5,
            n_max_gen=2,
        )

        # Mock the result
        mock_result = MagicMock()
        mock_result.X = np.array([[0.5], [1.5]])
        mock_minimize.return_value = mock_result

        _predictions, pareto_df = generator.generate_pareto_front()

        assert mock_minimize.called
        assert isinstance(pareto_df, pl.DataFrame)

    def test_pareto_front_generator_deterministic(self):
        """Test that ParetoFrontGenerator with same seed produces consistent results."""
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

        base_problem1 = SimpleProblem(
            investigation_space=space,
            optimization_problem=problem,
        )
        base_problem2 = SimpleProblem(
            investigation_space=space,
            optimization_problem=problem,
        )

        generator1 = ParetoFrontGenerator(problem=base_problem1, seed=42)
        generator2 = ParetoFrontGenerator(problem=base_problem2, seed=42)

        assert generator1.seed == generator2.seed
