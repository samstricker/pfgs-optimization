"""Initialization for the pareto_visualisation app module."""

from .html_layout import get_app_layout
from .server import app

# Import callbacks after app is fully initialized to avoid circular imports
from . import callbacks as _callbacks  # noqa: F401

__all__ = (
    "app",
    "get_app_layout",
)
