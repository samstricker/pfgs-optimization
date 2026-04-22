"""Data models for the investigation space."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .investigation_space_var import InvestigationSpaceVariable


@dataclass(frozen=True)
class InvestigationSpace:
    """Representation of the investigation space.

    Parameters
    ----------
    variables : List[InvestigationSpaceVariable]
        List of variables/dimensions in the investigation space.
    """

    variables: List[InvestigationSpaceVariable]

    @classmethod
    def from_json(cls, json_path: Path) -> "InvestigationSpace":
        """Create an InvestigationSpace instance from a JSON string.

        Parameters
        ----------
        json_path : Path
            The JSON string representing the investigation space.

        Returns
        -------
        InvestigationSpace
            The created InvestigationSpace instance.
        """
        data = json.loads(json_path.read_text())
        variables = [
            InvestigationSpaceVariable(
                name=var["name"],
                min=var["min"],
                max=var["max"],
                discretization=var["discretization"],
                expected_error=var["expected_error"],
            )
            for var in data.get("variables", [])
        ]
        return cls(variables=variables)

    def to_json(self, file_path: Path):
        """Save the InvestigationSpace instance to a JSON file.

        Parameters
        ----------
        file_path : Path
            The path to the JSON file to save the investigation space.
        """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=4)

    @property
    def labels(self) -> List[str]:
        """Get the labels (names) of all variables in the investigation space."""
        return [var.name for var in self.variables]

    @property
    def bounds(self) -> Tuple[List[float], List[float]]:
        """Get the bounds (min and max) of all variables in the investigation space.

        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing two lists: the first list contains the minimum bounds,
            and the second list contains the maximum bounds.
        """
        lower_bounds = [var.min for var in self.variables]
        upper_bounds = [var.max for var in self.variables]
        return lower_bounds, upper_bounds

    @property
    def expected_errors(self) -> List[Optional[float]]:
        """Get the expected errors of all variables in the investigation space."""
        return [var.expected_error for var in self.variables]

    @property
    def int_bounds(self) -> Tuple[List[int], List[int]]:
        """Get the integer bounds (min and max) of all variables using the discretization."""
        lower_int_bounds = []
        upper_int_bounds = []
        for var in self.variables:
            if var.discretization is None:
                raise ValueError(
                    f"Variable '{var.name}' does not have a discretization defined."
                )
            lower = int(np.ceil(var.min / var.discretization))
            upper = int(np.floor(var.max / var.discretization))
            lower_int_bounds.append(lower)
            upper_int_bounds.append(upper)
        return lower_int_bounds, upper_int_bounds

    @property
    def n_var(self) -> int:
        """Get the number of variables in the investigation space."""
        return len(self.variables)

    def recover_real_values(self, int_values: NDArray[np.int_]) -> NDArray[np.float64]:
        """Recover the real values from integer values using the discretization.

        Parameters
        ----------
        int_values : NDArray[np.int_]
            The integer values to recover.

        Returns
        -------
        NDArray[np.float64]
            The recovered real values.
        """
        real_values = np.zeros_like(int_values, dtype=np.float64)
        for i, var in enumerate(self.variables):
            if var.discretization is None:
                raise ValueError(
                    f"Variable '{var.name}' does not have a discretization defined."
                )
            real_values[:, i] = int_values[:, i] * var.discretization
        return real_values

    def round(self, real_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Round the real values to the nearest discretized values.

        Parameters
        ----------
        real_values : NDArray[np.float64]
            The real values to round.

        Returns
        -------
        NDArray[np.float64]
            The rounded values.
        """
        rounded_values = np.zeros_like(real_values, dtype=np.float64)
        for i, var in enumerate(self.variables):
            if var.discretization is None:
                raise ValueError(
                    f"Variable '{var.name}' does not have a discretization defined."
                )
            rounded_values[:, i] = (
                np.round(real_values[:, i] / var.discretization) * var.discretization
            )
        return rounded_values

    def as_dict(self):
        """Convert the InvestigationSpace instance to a dictionary."""
        return {
            "variables": [
                {
                    "name": var.name,
                    "min": var.min,
                    "max": var.max,
                    "discretization": var.discretization,
                    "expected_error": var.expected_error,
                }
                for var in self.variables
            ]
        }

    def create_meshgrid(
        self, points_per_dim: int
    ) -> Tuple[NDArray[np.float64], List[str]]:
        """Create a meshgrid of the investigation space for visualization.

        Returns
        -------
        NDArray[np.float64]
            The meshgrid data points.
        List[str]
            The labels of the variables.
        """

        grids = []
        for var in self.variables:
            grid = np.linspace(var.min, var.max, points_per_dim)
            grids.append(grid)
        mesh = np.meshgrid(*grids, indexing="ij")
        meshgrid_points = np.vstack([m.flatten() for m in mesh]).T
        return meshgrid_points, self.labels
