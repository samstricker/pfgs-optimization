"""This module initializes the package."""

from .data_models import (
    InvestigationSpace,
    InvestigationSpaceVariable,
    OptimizationAxis,
    OptimizationProblem,
)
from .sampling import BaseProblem, ParetoFrontGenerator

__all__ = [
    "InvestigationSpace",
    "InvestigationSpaceVariable",
    "OptimizationAxis",
    "OptimizationProblem",
    "BaseProblem",
    "ParetoFrontGenerator",
]
