"""Data models for the optimization problem."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List

from .optimization_axis import OptimizationAxis


@dataclass(frozen=True)
class OptimizationProblem:
    """Representation of an optimization problem. Multiple objectives to optimize."""

    problem: List[OptimizationAxis]

    @classmethod
    def from_json(cls, file_path: Path) -> "OptimizationProblem":
        """Create an OptimizationProblem instance from a dictionary."""
        data = json.loads(file_path.read_text())
        problem = [
            OptimizationAxis(
                var_name=axis["var_name"],
                minimize=axis["minimize"],
                optimization_type=axis["optimization_type"],
                lower_bound=axis["lower_bound"],
                upper_bound=axis["upper_bound"],
            )
            for axis in data.get("problem", [])
        ]
        return cls(problem=problem)

    @property
    def n_obj(self) -> int:
        """Get the number of objectives in the optimization problem."""
        return len(self.problem)

    def to_json(self, file_path: Path):
        """Save the OptimizationProblem instance to a JSON file.

        Parameters
        ----------
        file_path : Path
            The path to the JSON file to save the optimization problem.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=4)

    def __post_init__(self):
        if not isinstance(self.problem, list):
            raise TypeError(f"Problem must be a list, got {type(self.problem)}.")
        for axis in self.problem:
            if not isinstance(axis, OptimizationAxis):
                raise TypeError(
                    f"Each item in problem must be an OptimizationAxis, got {type(axis)}."
                )

    def as_dict(self):
        """Convert the OptimizationProblem instance to a dictionary."""
        return {
            "problem": [
                {
                    "var_name": axis.var_name,
                    "minimize": axis.minimize,
                    "optimization_type": axis.optimization_type,
                    "lower_bound": axis.lower_bound,
                    "upper_bound": axis.upper_bound,
                }
                for axis in self.problem
            ]
        }

    @property
    def objective_labels(self) -> List[str]:
        """Get the labels of the objectives in the optimization problem."""
        names = []
        for axis in self.problem:
            name = "obj:"
            if axis.minimize:
                name += "min("
            else:
                name += "max("
            if axis.optimization_type == "objective":
                name += f"μ({axis.var_name}))"
            elif axis.optimization_type == "constraint":
                name += f"P({axis.var_name}))"
            elif axis.optimization_type == "expected_improvement":
                name += f"EI({axis.var_name}))"
            elif axis.optimization_type == "standard_deviation":
                name += f"σ({axis.var_name}))"
            elif axis.optimization_type == "robustness":
                name += f"R({axis.var_name}))"
            else:
                name += f"{axis.var_name})"
            names.append(name)
        return names
