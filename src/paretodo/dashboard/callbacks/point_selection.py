"""Callbacks for point selection and clicking."""

import json
from pathlib import Path
import traceback

from dash import Input, Output, State, ALL, callback_context
import numpy as np
import pandas as pd

from paretodo.data_models import OptimizationProblem, InvestigationSpace

from ..server import app
from ..utils.types_and_constants import (
    ClickedPointsData,
    ObjectiveBoundsData,
    LOCAL_TEMP_DIR,
)
from ..utils.data_utils import load_pareto_df, load_experiments_df

__all__ = ["handle_all_clicks", "automated_selection", "reset_selection"]


@app.callback(
    Output("clicked-points-store", "data"),
    Input("pareto-front-graph", "clickData"),
    Input({"type": "experiment-results-graph", "index": ALL}, "clickData"),
    State("clicked-points-store", "data"),
    prevent_initial_call=False,
)
def handle_all_clicks(
    pareto_click_data,
    _experiment_click_data,
    store_data: ClickedPointsData,
) -> ClickedPointsData:
    """Handle click events and toggle point selection in store."""
    if not callback_context.triggered:
        return store_data

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    click_data = None

    if trigger_id == "pareto-front-graph":
        click_data = pareto_click_data
    elif (
        isinstance(trigger_id, dict)
        and trigger_id.get("type") == "experiment-results-graph"
    ):
        click_data = callback_context.triggered[0]["value"]
    else:
        if isinstance(trigger_id, str) and trigger_id.startswith("{"):
            try:
                parsed = json.loads(trigger_id)
                if parsed.get("type") == "experiment-results-graph":
                    click_data = callback_context.triggered[0]["value"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    if not click_data:
        return store_data

    if not store_data:
        store_data = {
            "investigation-space": [],
            "pareto-front": [],
            "experiment-results": [],
        }

    try:
        point_num = click_data["points"][0]["pointNumber"]

        try:
            raw_customdata = click_data["points"][0]["customdata"]
            row_index = int(raw_customdata[0])
        except (IndexError, TypeError, ValueError):
            row_index = point_num

        row_index = int(row_index)

        if trigger_id == "pareto-front-graph":
            current_list = store_data.get("pareto-front", [])
            if row_index in current_list:
                current_list.remove(row_index)
            else:
                current_list.append(row_index)
            store_data["pareto-front"] = current_list
            store_data["investigation-space"] = current_list
        else:
            current_list = store_data.get("experiment-results", [])
            if row_index in current_list:
                current_list.remove(row_index)
            else:
                current_list.append(row_index)
            store_data["experiment-results"] = current_list

        return store_data
    except (KeyError, IndexError, TypeError, ValueError):
        return store_data


@app.callback(
    Output("pinned-points-store", "data"),
    Input("automated-selection-button", "n_clicks"),
    State("mlflow-run", "value"),
    State("iteration", "value"),
    State("objectives-bounds-store", "data"),
    Input("batch-size-input", "value"),
    prevent_initial_call=True,
)
def automated_selection(
    n_clicks: int | None,
    mlflow_run: str | None,
    iteration: str | None,
    objectives_bounds: ObjectiveBoundsData,
    batch_size: int | None,
) -> dict[str, bool]:
    """Automated diversity-based selection of Pareto points.

    Selects batch_size points from the Pareto front that are within the
    specified objective bounds (non-restricted points only). Uses diversity-based
    greedy selection:
    - First point: closest to Utopia point (minimum for each objective)
    - Subsequent points: maximize minimum distance to previously selected points
      and all experiment data points

    Parameters
    ----------
    n_clicks : int | None
        Button click count
    mlflow_run : str | None
        MLflow run ID
    iteration : str | None
        Iteration number
    objectives_bounds : ObjectiveBoundsData
        Objective bounds filter
    batch_size : int | None
        Number of points to select

    Returns
    -------
    dict[str, bool]
        Dictionary mapping point indices to True for pinned/selected points
    """
    if not n_clicks or mlflow_run is None or iteration is None:
        print("Early return: missing required inputs")
        return {}

    try:
        # Load Pareto front and experiments data
        pareto_df = load_pareto_df(mlflow_run, iteration)
        experiments_df = load_experiments_df(mlflow_run, iteration)

        # Load optimization problem to get minimize/maximize info for each objective
        opt_prob_path = Path(
            f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/optimization_problem.json"
        )
        optimization_problem = OptimizationProblem.from_json(opt_prob_path)

        # Load investigation space to get variable names
        inv_space_path = Path(
            f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/investigation_space.json"
        )
        investigation_space = InvestigationSpace.from_json(inv_space_path)

        # Get investigation space variable names for distance calculations in subsequent iterations
        investigation_var_names = [var.name for var in investigation_space.variables]

        # Create mapping from objective var_name to minimize flag
        minimize_map = {}
        for axis in optimization_problem.problem:
            # The objective column name is "obj:{axis.var_name}"
            obj_col_name = f"obj:{axis.var_name}"
            minimize_map[obj_col_name] = axis.minimize

        # Extract objective columns
        objective_columns = [col for col in pareto_df.columns if col.startswith("obj:")]

        if not objective_columns:
            print("No objective columns found in pareto_df")
            return {}

        batch_size = int(max(1, min(batch_size or 5, len(pareto_df))))

        # Normalize objectives for distance calculation
        objectives_data = pareto_df[objective_columns].copy()
        objectives_data_normalized = objectives_data.copy()
        for col in objective_columns:
            col_min = objectives_data[col].min()
            col_max = objectives_data[col].max()
            if col_max > col_min:
                objectives_data_normalized[col] = (objectives_data[col] - col_min) / (
                    col_max - col_min
                )

        # Filter out points outside objective bounds (keep only non-restricted points)
        remaining_indices = set(range(len(pareto_df)))

        if objectives_bounds:
            valid_indices = set()
            for idx in remaining_indices:
                is_outside_bounds = False
                for obj_col, bounds in objectives_bounds.items():
                    if obj_col in pareto_df.columns:
                        value = pareto_df.loc[idx, obj_col]

                        # Convert value to float for comparison
                        try:
                            # Skip complex numbers
                            if isinstance(value, complex):
                                continue
                            value_float = float(value)  # type: ignore
                        except (ValueError, TypeError):
                            continue

                        min_bound = bounds.get("min")
                        max_bound = bounds.get("max")

                        if min_bound is not None:
                            try:
                                if value_float < float(min_bound):
                                    is_outside_bounds = True
                                    break
                            except (ValueError, TypeError):
                                pass

                        if max_bound is not None:
                            try:
                                if value_float > float(max_bound):
                                    is_outside_bounds = True
                                    break
                            except (ValueError, TypeError):
                                pass

                if not is_outside_bounds:
                    valid_indices.add(idx)
            remaining_indices = valid_indices

        if not remaining_indices:
            print("No remaining indices after bounds filtering")
            return {}

        # Calculate Utopia point based only on valid points
        # For minimization objectives: use minimum value
        # For maximization objectives: use maximum value
        valid_objectives = objectives_data_normalized.iloc[list(remaining_indices)][
            objective_columns
        ]
        utopia_point_values = []

        for obj_col in objective_columns:
            is_minimize = minimize_map.get(
                obj_col, True
            )  # Default to minimize if not found
            if is_minimize:
                # For minimization: Utopia is at minimum (0 in normalized space typically)
                value = valid_objectives[obj_col].min()
            else:
                # For maximization: Utopia is at maximum (1 in normalized space typically)
                value = valid_objectives[obj_col].max()
            utopia_point_values.append(value)

        utopia_point = np.asarray(utopia_point_values)

        # Normalize experiments data for distance calculation
        # Check which objective columns exist in experiments_df
        available_obj_cols = [
            col for col in objective_columns if col in experiments_df.columns
        ]

        if not available_obj_cols:
            # If no objective columns in experiments, use measurement columns instead
            available_obj_cols = experiments_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()[: len(objective_columns)]

        experiments_objectives = (
            experiments_df[available_obj_cols].copy()
            if available_obj_cols
            else pd.DataFrame()
        )

        # Normalize experiments data
        for col in experiments_objectives.columns:
            if col in objectives_data.columns:
                col_min = objectives_data[col].min()
                col_max = objectives_data[col].max()
                if col_max > col_min:
                    experiments_objectives[col] = (
                        experiments_objectives[col] - col_min
                    ) / (col_max - col_min)

        selected_indices = []

        # First point: closest to Utopia point
        best_idx = None
        best_distance = float("inf")
        for candidate_idx in remaining_indices:
            candidate_point = np.asarray(
                objectives_data_normalized.iloc[candidate_idx].values
            )
            distance = np.sqrt(np.sum((candidate_point - utopia_point) ** 2))
            if distance < best_distance:
                best_distance = distance
                best_idx = candidate_idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.discard(best_idx)

        # For subsequent points, use investigation space variables for diversity
        # Extract investigation space variables and normalize them
        # Get investigation space variables from pareto_df
        pareto_inv_space = pareto_df[investigation_var_names].copy()

        # Get investigation space variables from experiments_df (if available)
        experiments_inv_space = experiments_df[
            [col for col in investigation_var_names if col in experiments_df.columns]
        ].copy()

        pareto_inv_space_normalized = pareto_inv_space.copy()
        for col in investigation_var_names:
            if col in pareto_df.columns and col in experiments_df.columns:
                # Use combined min/max
                all_values = pd.concat(
                    [pareto_df[[col]], experiments_df[[col]]], ignore_index=True
                )
                col_min = all_values[col].min()
                col_max = all_values[col].max()
            elif col in pareto_df.columns:
                col_min = pareto_df[col].min()
                col_max = pareto_df[col].max()
            else:
                continue

            if col_max > col_min:
                pareto_inv_space_normalized[col] = (pareto_df[col] - col_min) / (
                    col_max - col_min
                )

        # Normalize experiments investigation space
        experiments_inv_space_normalized = experiments_inv_space.copy()
        for col in experiments_inv_space.columns:
            if col in pareto_df.columns and col in experiments_df.columns:
                all_values = pd.concat(
                    [pareto_df[[col]], experiments_df[[col]]], ignore_index=True
                )
                col_min = all_values[col].min()
                col_max = all_values[col].max()
            elif col in pareto_df.columns:
                col_min = pareto_df[col].min()
                col_max = pareto_df[col].max()
            else:
                continue

            if col_max > col_min:
                experiments_inv_space_normalized[col] = (
                    experiments_df[col] - col_min
                ) / (col_max - col_min)

        for _ in range(batch_size - 1):
            if not remaining_indices:
                break

            best_idx = None
            best_min_distance = -1.0

            for candidate_idx in remaining_indices:
                candidate_point = np.asarray(
                    pareto_inv_space_normalized.iloc[candidate_idx].values
                )
                min_distance = float("inf")

                # Check distance to all previously selected points (in investigation space)
                for selected_idx in selected_indices:
                    selected_point = np.asarray(
                        pareto_inv_space_normalized.iloc[selected_idx].values
                    )
                    distance = np.sqrt(np.sum((candidate_point - selected_point) ** 2))
                    min_distance = min(min_distance, distance)

                # Check distance to all experiment data points (in investigation space)
                if not experiments_inv_space_normalized.empty:
                    for exp_idx in range(len(experiments_inv_space_normalized)):
                        exp_point = np.asarray(
                            experiments_inv_space_normalized.iloc[exp_idx].values
                        )
                        if len(exp_point) > 0:  # Only if exp_point has values
                            distance = np.sqrt(
                                np.sum((candidate_point - exp_point) ** 2)
                            )
                            min_distance = min(min_distance, distance)

                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_idx = candidate_idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.discard(best_idx)

        # Convert to pinned data format
        result = {str(idx): True for idx in selected_indices}
        return result

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
    ) as e:
        print(f"Error in automated_selection: {type(e).__name__}: {e}")

        traceback.print_exc()
        return {}


@app.callback(
    Output("pinned-points-store", "data", allow_duplicate=True),
    Input("reset-selection-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_selection(n_clicks: int | None) -> dict:
    """Reset all pinned point selections.

    Parameters
    ----------
    n_clicks : int | None
        Button click count

    Returns
    -------
    dict
        Empty dictionary (no pinned points)
    """
    if not n_clicks:
        return {}
    return {}
