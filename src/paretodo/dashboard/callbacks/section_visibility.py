"""Callback: mode-dependent section visibility.

Reads the selected run's MLflow ``mode`` tag and hides sections that are not
relevant to that mode. The tag is set by run_bo.py:

    default             -> show all sections
    candidate_selection -> hide Experiment Results (the pairwise scatter grid),
                           GP Predictions, and Investigation Space Visualization

Those three assume a navigable design space: they plot pairs of
investigation-space variables, or a grid over them. Candidate selection picks
from a fixed pool instead, so the axes are a surrogate representation rather
than knobs a user tunes, and the panels answer the wrong question. The
Experiment Results *parallel coordinates* plot is a separate section and stays
visible.

A missing tag (older runs) is treated as default, so nothing is hidden.
"""

from dash import Input, Output

import mlflow

from ..server import app

__all__ = ["update_section_visibility"]

SHOWN = {"display": "block"}
HIDDEN = {"display": "none"}


def _get_run_mode(mlflow_port, mlflow_experiment, mlflow_run):
    """Return the run's mode tag, defaulting to 'default'."""
    if mlflow_port is None or mlflow_experiment is None or mlflow_run is None:
        return "default"
    try:
        client = mlflow.tracking.MlflowClient(
            tracking_uri=f"http://127.0.0.1:{mlflow_port}"
        )
        experiment = client.get_experiment_by_name(mlflow_experiment)
        if experiment is None:
            return "default"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName='{mlflow_run}'",
        )
        if not runs:
            return "default"
        return runs[0].data.tags.get("mode", "default")
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error reading run mode: {e}")
        return "default"


@app.callback(
    Output("experiment-results-section", "style"),
    Output("gp-predictions-section", "style"),
    Output("investigation-space-viz-section", "style"),
    Input("mlflow-port", "value"),
    Input("mlflow-experiment", "value"),
    Input("mlflow-run", "value"),
)
def update_section_visibility(mlflow_port, mlflow_experiment, mlflow_run):
    """Hide design-space sections in candidate_selection mode."""
    mode = _get_run_mode(mlflow_port, mlflow_experiment, mlflow_run)

    if mode == "candidate_selection":
        return HIDDEN, HIDDEN, HIDDEN
    return SHOWN, SHOWN, SHOWN