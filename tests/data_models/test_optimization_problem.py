"""Tests for OptimizationAxis and OptimizationProblem classes."""

import json
import tempfile
from pathlib import Path

import pytest

from paretodo import OptimizationAxis, OptimizationProblem


class TestOptimizationProblem:
    """Tests for OptimizationProblem class."""

    def test_valid_optimization_problem_creation(self):
        """Test creating a valid OptimizationProblem."""
        axis1 = OptimizationAxis(
            var_name="temperature",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        axis2 = OptimizationAxis(
            var_name="pressure",
            minimize=False,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis1, axis2])
        assert len(problem.problem) == 2
        assert problem.n_obj == 2

    def test_invalid_problem_type(self):
        """Test that non-list problem raises TypeError."""
        with pytest.raises(TypeError, match="Problem must be a list"):
            OptimizationProblem(problem="not a list")

    def test_invalid_problem_item_type(self):
        """Test that non-OptimizationAxis items raise TypeError."""
        with pytest.raises(
            TypeError, match="Each item in problem must be an OptimizationAxis"
        ):
            OptimizationProblem(problem=["not an axis"])

    def test_n_obj_property(self):
        """Test n_obj property returns correct number of objectives."""
        axis1 = OptimizationAxis(
            var_name="temperature",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        axis2 = OptimizationAxis(
            var_name="pressure",
            minimize=False,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis1, axis2])
        assert problem.n_obj == 2

    def test_to_json(self):
        """Test saving OptimizationProblem to JSON file."""
        axis1 = OptimizationAxis(
            var_name="temperature",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis1])

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "problem.json"
            problem.to_json(json_path)

            assert json_path.exists()
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            assert len(data["problem"]) == 1
            assert data["problem"][0]["var_name"] == "temperature"
            assert data["problem"][0]["minimize"] is True
            assert data["problem"][0]["optimization_type"] == "objective"

    def test_from_json(self):
        """Test loading OptimizationProblem from JSON file."""
        problem_data = {
            "problem": [
                {
                    "var_name": "temperature",
                    "minimize": True,
                    "optimization_type": "objective",
                    "lower_bound": None,
                    "upper_bound": None,
                },
                {
                    "var_name": "pressure",
                    "minimize": False,
                    "optimization_type": "constraint",
                    "lower_bound": 1.0,
                    "upper_bound": 10.0,
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "problem.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(problem_data, f)

            problem = OptimizationProblem.from_json(json_path)

            assert len(problem.problem) == 2
            assert problem.problem[0].var_name == "temperature"
            assert problem.problem[0].minimize is True
            assert problem.problem[1].var_name == "pressure"
            assert problem.problem[1].lower_bound == 1.0
            assert problem.problem[1].upper_bound == 10.0

    def test_as_dict(self):
        """Test converting OptimizationProblem to dictionary."""
        axis1 = OptimizationAxis(
            var_name="temperature",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis1])
        result = problem.as_dict()

        assert "problem" in result
        assert len(result["problem"]) == 1
        assert result["problem"][0]["var_name"] == "temperature"

    def test_objective_labels_simple(self):
        """Test objective_labels property with simple objectives."""
        axis1 = OptimizationAxis(
            var_name="temperature",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis1])
        labels = problem.objective_labels

        assert len(labels) == 1
        assert "min" in labels[0]
        assert "temperature" in labels[0]

    def test_objective_labels_constraint(self):
        """Test objective_labels property with constraint optimization type."""
        axis1 = OptimizationAxis(
            var_name="temperature",
            minimize=False,
            optimization_type="constraint",
            lower_bound=50.0,
            upper_bound=100.0,
        )
        problem = OptimizationProblem(problem=[axis1])
        labels = problem.objective_labels

        assert len(labels) == 1
        assert "max" in labels[0]
        assert "P(" in labels[0]

    def test_objective_labels_ei(self):
        """Test objective_labels property with expected_improvement type."""
        axis1 = OptimizationAxis(
            var_name="temperature",
            minimize=True,
            optimization_type="expected_improvement",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis1])
        labels = problem.objective_labels

        assert len(labels) == 1
        assert "EI(" in labels[0]

    def test_objective_labels_std_dev(self):
        """Test objective_labels property with standard_deviation type."""
        axis1 = OptimizationAxis(
            var_name="temperature",
            minimize=True,
            optimization_type="standard_deviation",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis1])
        labels = problem.objective_labels

        assert len(labels) == 1
        assert "σ(" in labels[0]

    def test_objective_labels_robustness(self):
        """Test objective_labels property with robustness type."""
        axis1 = OptimizationAxis(
            var_name="temperature",
            minimize=False,
            optimization_type="robustness",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis1])
        labels = problem.objective_labels

        assert len(labels) == 1
        assert "R(" in labels[0]

    def test_problem_is_frozen(self):
        """Test that OptimizationProblem is immutable."""
        axis1 = OptimizationAxis(
            var_name="temperature",
            minimize=True,
            optimization_type="objective",
            lower_bound=None,
            upper_bound=None,
        )
        problem = OptimizationProblem(problem=[axis1])
        with pytest.raises(AttributeError):
            problem.problem = []
