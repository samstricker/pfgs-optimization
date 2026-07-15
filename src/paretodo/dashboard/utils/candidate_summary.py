"""Helpers for candidate-selection mode.

Candidate selection picks from a fixed pool of existing candidates rather than
navigating a design space, so the investigation-space table (one row per
surrogate feature) is noise. In that mode the panel shows a short summary
instead: the size of the candidate pool and how many have been screened at the
selected iteration.

Pool size comes from the run's ``n_candidates`` param, logged by run_bo.py.
Screened count is the row count of the iteration's experiments.csv, which holds
the observed set and grows by one each iteration.
"""

import os

from dash import html
import mlflow
import pandas as pd

from .types_and_constants import LOCAL_TEMP_DIR

__all__ = ["get_run_mode", "build_candidate_summary"]


def get_run(mlflow_port, mlflow_experiment, mlflow_run):
    """Return the MLflow Run object, or None."""
    if mlflow_port is None or mlflow_experiment is None or mlflow_run is None:
        return None
    try:
        client = mlflow.tracking.MlflowClient(
            tracking_uri=f"http://127.0.0.1:{mlflow_port}"
        )
        experiment = client.get_experiment_by_name(mlflow_experiment)
        if experiment is None:
            return None
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName='{mlflow_run}'",
        )
        return runs[0] if runs else None
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error fetching run: {e}")
        return None


def get_run_mode(mlflow_port, mlflow_experiment, mlflow_run):
    """Return the run's mode tag. Missing tag (older runs) means 'default'."""
    run = get_run(mlflow_port, mlflow_experiment, mlflow_run)
    if run is None:
        return "default"
    return run.data.tags.get("mode", "default")


def _screened_count(mlflow_run, iteration):
    """Rows in the iteration's experiments.csv, i.e. molecules screened so far."""
    if iteration is None:
        return None
    path = os.path.join(LOCAL_TEMP_DIR, mlflow_run, iteration, "experiments.csv")
    try:
        return len(pd.read_csv(path))
    except (FileNotFoundError, ValueError, OSError):
        return None


def build_candidate_summary(mlflow_port, mlflow_experiment, mlflow_run, iteration):
    """Summary panel shown in place of the investigation-space table."""
    run = get_run(mlflow_port, mlflow_experiment, mlflow_run)

    pool = None
    if run is not None:
        raw = run.data.params.get("n_candidates")
        if raw is not None:
            try:
                pool = int(raw)
            except ValueError:
                pool = None

    screened = _screened_count(mlflow_run, iteration)

    def row(label, value):
        return html.Tr(
            [
                html.Td(label, style={"fontWeight": "bold", "paddingRight": "20px"}),
                html.Td(value),
            ]
        )

    rows = [row("Mode", "Candidate Selection")]

    if pool is not None:
        rows.append(row("Candidate pool", f"{pool} molecules"))
    else:
        rows.append(row("Candidate pool", "unknown"))

    if screened is not None and pool is not None:
        rows.append(row("Screened so far", f"{screened} of {pool}"))
    elif screened is not None:
        rows.append(row("Screened so far", f"{screened}"))
    else:
        rows.append(row("Screened so far", "unknown"))

    return html.Div(
        [
            html.Table(rows, style={"borderCollapse": "collapse"}),
        ]
    )
