"""Data models for the investigation space."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InvestigationSpaceVariable:
    """Representation of a variable/dimension in the investigation space.

    Parameters
    ----------
    name : str
        The name of the variable.
    min : float
        The minimum value of the variable.
    max : float
        The maximum value of the variable.
    discretization : Optional[float]
        The discretization level of the variable, by default None.
    expected_error : Optional[float]
        The expected error for the variable, by default None.
    """

    name: str
    min: float
    max: float
    discretization: Optional[float] = None
    expected_error: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(f"Variable name must be a string, got {type(self.name)}.")
        if not isinstance(self.min, float):
            raise TypeError(
                f"Variable '{self.name}': min must be a number, got {type(self.min)}."
            )
        if not isinstance(self.max, float):
            raise TypeError(
                f"Variable '{self.name}': max must be a number, got {type(self.max)}."
            )
        if self.discretization is not None and not isinstance(
            self.discretization, float
        ):
            raise TypeError(
                f"Variable '{self.name}': discretization must be a number, "
                f"got {type(self.discretization)}."
            )
        if self.min >= self.max:
            raise ValueError(
                f"Variable '{self.name}': min ({self.min}) must be less than max ({self.max})."
            )
        if self.discretization is not None and self.discretization < 0:
            raise ValueError(
                f"Variable '{self.name}': discretization ({self.discretization}) must be positive."
            )
        if self.expected_error is not None:
            if not isinstance(self.expected_error, float):
                raise TypeError(
                    f"Variable '{self.name}': expected_error must be a number, "
                    f"got {type(self.expected_error)}."
                )
            if self.expected_error < 0:
                raise ValueError(
                    f"Variable '{self.name}': expected_error ({self.expected_error})"
                    "must be non-negative."
                )
            if self.expected_error >= self.max - self.min:
                raise ValueError(
                    f"Variable '{self.name}': expected_error ({self.expected_error}) "
                    f"must be less than the range (max - min = {self.max - self.min})."
                )
