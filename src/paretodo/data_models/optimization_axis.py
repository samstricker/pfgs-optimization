"""Data models for the optimization problem."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizationAxis:
    """Representation of an optimization axis. One of objective to optimize."""

    var_name: str
    minimize: bool
    optimization_type: str
    lower_bound: Optional[float]
    upper_bound: Optional[float]

    def __post_init__(self):
        if not isinstance(self.var_name, str):
            raise TypeError(
                f"Variable name must be a string, got {type(self.var_name)}."
            )
        if not isinstance(self.minimize, bool):
            raise TypeError(
                f"Minimize flag must be a boolean, got {type(self.minimize)}."
            )
        if not isinstance(self.optimization_type, str):
            raise TypeError(
                f"Optimization type must be a string, got {type(self.optimization_type)}."
            )
        if self.lower_bound is not None and not isinstance(self.lower_bound, float):
            raise TypeError(
                f"Lower bound must be a float or None, got {type(self.lower_bound)}."
            )
        if self.upper_bound is not None and not isinstance(self.upper_bound, float):
            raise TypeError(
                f"Upper bound must be a float or None, got {type(self.upper_bound)}."
            )
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound >= self.upper_bound
        ):
            raise ValueError(
                f"Lower bound ({self.lower_bound}) must be less than upper bound "
                f"({self.upper_bound})."
            )

    def set_minimize(self, minimize: bool):
        """Set the minimize flag for this optimization axis.

        Parameters
        ----------
        minimize : bool
            Whether to minimize (True) or maximize (False) this optimization axis.
        """
        self.minimize = minimize
