"""Callbacks for loading data from MLflow into the Pareto visualization app."""

import os
from pathlib import Path

from dash import Input, Output, html
import mlflow
import pandas as pd

from paretodo.data_models import InvestigationSpace

from ..server import app

from ..utils.helper_functions import get_client_and_run
from ..utils.types_and_constants import LOCAL_TEMP_DIR
from ..utils.data_utils import load_optimization_problem
from ..utils.table_components import (
    create_investigation_space_table,
    create_optimization_problem_table,
    create_measurements_table,
)


@app.callback(
    Output("mlflow-experiment", "options"),
    Input("mlflow-port", "value"),
)
def update_mlflow_experiments_dropdown(mlflow_port):
    """Update the MLflow experiments dropdown based on the selected port.

    Parameters
    ----------
    mlflow_port : int
        The MLflow port number.

    Returns
    -------
    list
        List of experiment names.
    """
    if mlflow_port is None:
        return []
    try:
        client = mlflow.tracking.MlflowClient(
            tracking_uri=f"http://127.0.0.1:{mlflow_port}"
        )
        experiment_list = client.search_experiments()
        return [exp.name for exp in experiment_list]
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error fetching experiments: {e}")
        return []


@app.callback(
    Output("mlflow-run", "options"),
    Output("refresh-runs-button", "n_clicks"),
    Input("mlflow-port", "value"),
    Input("mlflow-experiment", "value"),
    Input("refresh-runs-button", "n_clicks"),
)
def update_mlflow_runs_dropdown(
    mlflow_port, mlflow_experiment, n_clicks
):  # pylint: disable=unused-argument
    """Update the MLflow runs dropdown based on selected experiment.

    This callback only fetches and displays run names without downloading artifacts.
    The n_clicks parameter serves as a trigger for the refresh button.

    Parameters
    ----------
    mlflow_port : int
        The MLflow port number.
    mlflow_experiment : str
        The selected experiment name.
    n_clicks : int
        Number of times the refresh button was clicked (trigger parameter).

    Returns
    -------
    tuple
        Tuple of (list of run names, reset n_clicks to 0).
    """
    if mlflow_experiment is None:
        return [], 0

    try:
        client = mlflow.tracking.MlflowClient(
            tracking_uri=f"http://127.0.0.1:{mlflow_port}"
        )
        mlflow_experiment_id = client.get_experiment_by_name(
            mlflow_experiment
        ).experiment_id
        run_list = client.search_runs(experiment_ids=[mlflow_experiment_id])
        experiment_list = [run.info.run_name for run in run_list]
        return experiment_list, 0
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error fetching runs: {e}")
        return [], 0


@app.callback(
    Output("iteration", "options"),
    Output("iteration", "value"),
    Input("mlflow-port", "value"),
    Input("mlflow-experiment", "value"),
    Input("mlflow-run", "value"),
    Input("refresh-iterations-button", "n_clicks"),
)
def update_iterations_and_download_artifacts(
    mlflow_port, mlflow_experiment, mlflow_run, refresh_clicks
):  # pylint: disable=unused-argument
    """Update iteration dropdown and download selected run artifacts.

    This callback handles two scenarios:
    1. When a run is selected: downloads artifacts and updates iterations
    2. When refresh button is clicked: re-downloads artifacts and updates iterations

    The refresh_clicks parameter serves as a trigger for the refresh button.

    Parameters
    ----------
    mlflow_port : int
        The MLflow port number.
    mlflow_experiment : str
        The selected experiment name.
    mlflow_run : str
        The selected run name.
    refresh_clicks : int
        Number of times the refresh button was clicked (trigger parameter).

    Returns
    -------
    tuple
        Tuple of (list of iteration options, default iteration value).
    """
    if mlflow_experiment is None or mlflow_run is None:
        return [], None

    try:
        client = mlflow.tracking.MlflowClient(
            tracking_uri=f"http://127.0.0.1:{mlflow_port}"
        )
        mlflow_experiment_id = client.get_experiment_by_name(
            mlflow_experiment
        ).experiment_id
        run_list = client.search_runs(experiment_ids=[mlflow_experiment_id])

        # Find the selected run
        selected_run = None
        for run in run_list:
            if run.info.run_name == mlflow_run:
                selected_run = run
                break

        if selected_run is None:
            return [], None

        # Create temp directory and download artifacts for selected run only
        temp_dir = os.path.join(LOCAL_TEMP_DIR, selected_run.info.run_name)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        try:
            # Download general artifacts
            client.download_artifacts(
                run_id=selected_run.info.run_id,
                path="general",
                dst_path=temp_dir,
            )
            # Discover and download all available iterations
            try:
                artifacts = client.list_artifacts(run_id=selected_run.info.run_id)
                for artifact in artifacts:
                    # Check if this is a directory (iteration folders are directories)
                    if artifact.is_dir and artifact.path.startswith("Iteration_"):
                        client.download_artifacts(
                            run_id=selected_run.info.run_id,
                            path=artifact.path,
                            dst_path=temp_dir,
                        )
            except (mlflow.exceptions.MlflowException, OSError) as e:
                print(
                    f"Error discovering iterations for run {selected_run.info.run_id}: {e}"
                )
        except (mlflow.exceptions.MlflowException, OSError) as e:
            print(
                f"Error downloading artifacts for run {selected_run.info.run_id}: {e}"
            )

        # Get available iterations from downloaded artifacts
        if not os.path.exists(temp_dir):
            return [], None

        options = [
            option.name
            for option in Path(temp_dir).iterdir()
            if option.is_dir() and option.name.startswith("Iteration_")
        ]
        if not options:
            return [], None
        options.sort()
        return options, options[0] if len(options) > 0 else None
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error updating iterations: {e}")
        return [], None


@app.callback(
    Input("mlflow-port", "value"),
    Input("mlflow-experiment", "value"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
)
def download_json_files(mlflow_port, mlflow_experiment, mlflow_run, iteration):
    """Download JSON files from MLflow artifacts.

    Parameters
    ----------
    mlflow_port : int
        The MLflow port number.
    mlflow_experiment : str
        The selected experiment name.
    mlflow_run : str
        The selected run name.
    iteration : str
        The selected iteration.
    """
    if iteration is None:
        return

    try:
        client, run = get_client_and_run(mlflow_port, mlflow_experiment, mlflow_run)
        if run is None or client is None:
            return
        temp_dir = os.path.join(LOCAL_TEMP_DIR, mlflow_run)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        client.download_artifacts(
            run_id=run.info.run_id,
            path="general",
            dst_path=temp_dir,
        )
        client.download_artifacts(
            run_id=run.info.run_id,
            path=iteration,
            dst_path=temp_dir,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error downloading JSON files: {e}")


@app.callback(
    Output("investigation-space-table", "children"),
    Output("optimization-problem-table", "children"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    Input("investigation-space-renames-store", "data"),
    Input("measurement-renames-store", "data"),
    Input("pareto-mode-store", "data"),
)
def update_investigation_optimization_tables(
    mlflow_run,
    iteration,
    investigation_space_renames,
    measurement_renames,
    pareto_mode,
):
    """Update the investigation space and optimization problem tables.

    Parameters
    ----------
    mlflow_run : str
        The selected run name.
    iteration : str
        The selected iteration.
    investigation_space_renames : dict
        Dictionary mapping investigation space variable names to their renamed values.
    measurement_renames : dict
        Dictionary mapping measurement names to their renamed values.

    Returns
    -------
    tuple
        Tuple of (investigation space + measurements table, optimization problem table).
    """
    if mlflow_run is None:
        return None, None

    try:
        investigation_space_data = InvestigationSpace.from_json(
            Path(f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/investigation_space.json")
        )
        investigation_space_table = create_investigation_space_table(
            investigation_space_data, investigation_space_renames
        )
    except FileNotFoundError:
        investigation_space_table = "Investigation space data not found."
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error loading investigation space: {e}")
        investigation_space_table = f"Error loading investigation space: {e}"

    measurements_table = html.Div()
    if iteration is not None:
        try:
            experiments_csv = (
                f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/experiments.csv"
            )
            experiments_df = pd.read_csv(experiments_csv)
            investigation_space_var_names = [
                var.name for var in investigation_space_data.variables
            ]
            measurements_table = html.Div(
                [
                    html.H3("Measurements", style={"marginTop": "20px"}),
                    create_measurements_table(
                        investigation_space_var_names,
                        experiments_df,
                        measurement_renames,
                    ),
                ]
            )
        except FileNotFoundError:
            measurements_table = html.Div()
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Error loading measurements: {e}")
            measurements_table = html.Div()

    try:
        optimization_problem_data = load_optimization_problem(mlflow_run, pareto_mode)
        optimization_problem = create_optimization_problem_table(
            optimization_problem_data, mlflow_run, iteration, pareto_mode
        )
    except FileNotFoundError:
        optimization_problem = "Optimization problem data not found."
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error loading optimization problem: {e}")
        optimization_problem = f"Error loading optimization problem: {e}"

    combined_investigation = html.Div(
        [
            investigation_space_table,
            measurements_table,
        ]
    )
    return combined_investigation, optimization_problem


@app.callback(
    Output("clicked-points-store", "data", allow_duplicate=True),
    Output("pinned-points-store", "data", allow_duplicate=True),
    Input("iteration", "value"),
    prevent_initial_call=True,
)
def reset_selections_on_iteration_change(iteration):
    """Reset both manual and automated selections when iteration changes.

    Parameters
    ----------
    iteration : str
        The selected iteration.

    Returns
    -------
    tuple
        Tuple of (reset clicked-points-store, reset pinned-points-store).
    """
    if iteration is None:
        return {
            "investigation-space": [],
            "pareto-front": [],
            "experiment-results": [],
        }, {}

    # Return reset values for both stores
    return (
        {
            "investigation-space": [],
            "pareto-front": [],
            "experiment-results": [],
        },
        {},
    )


@app.callback(
    Output("pareto-mode-store", "data"),
    Input("pareto-mode", "value"),
)
def update_pareto_mode(mode):
    """Update the pareto mode store when the mode dropdown changes.

    Parameters
    ----------
    mode : str
        The selected mode ("Optimization" or "Manufacturing").

    Returns
    -------
    str
        The selected mode.
    """
    return mode if mode is not None else "Optimization"
