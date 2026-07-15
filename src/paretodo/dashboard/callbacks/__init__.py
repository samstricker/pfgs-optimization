"""Main entry point for Pareto visualization app callbacks.

This module imports all callback functions and ensures they are registered
with the Dash app instance.
"""

from ..theme.plotly_theme import register_theme
from . import (
    point_selection,
    state_management,
    viewport,
    data_loading,
    experiment_results,
    pareto_front,
    gp_prediction,
    investigation_space,
    export_download,
    molecule_preview,
    top_ten_observed,
    section_visibility,
)
from .data_loading import (
    update_mlflow_experiments_dropdown,
    update_mlflow_runs_dropdown,
    update_iterations_and_download_artifacts,
    download_json_files,
    update_investigation_optimization_tables,
)

__all__ = []

register_theme()

__all__.extend(point_selection.__all__)
__all__.extend(state_management.__all__)
__all__.extend(viewport.__all__)
__all__.extend(experiment_results.__all__)
__all__.extend(pareto_front.__all__)
__all__.extend(gp_prediction.__all__)
__all__.extend(investigation_space.__all__)
__all__.extend(export_download.__all__)
__all__.extend(molecule_preview.__all__)
__all__.extend(top_ten_observed.__all__)
__all__.extend(section_visibility.__all__)
__all__.extend(
    [
        "update_mlflow_experiments_dropdown",
        "update_mlflow_runs_dropdown",
        "update_iterations_and_download_artifacts",
        "download_json_files",
        "update_investigation_optimization_tables",
    ]
)
