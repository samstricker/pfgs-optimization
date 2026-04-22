"""This module initializes the sampling subpackage."""

from .base_problem import BaseProblem
from .pareto_front_generator import ParetoFrontGenerator

__all__ = ["BaseProblem", "ParetoFrontGenerator"]
