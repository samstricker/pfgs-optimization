"""Callbacks for point selection and clicking."""

import json
from pathlib import Path
import traceback

from dash import Input, Output, State, ALL, callback_context
import polars as pl

from paretodo.data_models import OptimizationProblem, InvestigationSpace
from paretodo.selection.selector import ParetoSelector

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
    """Automated diversity-based selection of Pareto points using ParetoSelector.

    Selects batch_size points from the Pareto front that are within the
    specified objective bounds (non-restricted points only). Uses diversity-based
    greedy selection:
    - First point: closest to Utopia point (minimum for each objective)
    - Subsequent points: maximize minimum distance to previously selected points

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

        # Extract objective columns
        objective_columns = [col for col in pareto_df.columns if col.startswith("obj:")]

        if not objective_columns:
            return {}

        batch_size = int(max(1, min(batch_size or 5, len(pareto_df))))

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
            return {}

        # Create filtered pareto dataframe with original indices tracked
        pareto_filtered = pareto_df.iloc[list(remaining_indices)].reset_index(drop=True)
        pareto_filtered_pl = pl.from_pandas(pareto_filtered)

        # Add column to track original indices
        original_remaining_indices = list(remaining_indices)
        pareto_filtered_pl = pareto_filtered_pl.with_row_count(
            "__original_idx"
        ).with_columns(
            pl.col("__original_idx").map_elements(
                lambda i: original_remaining_indices[i], return_dtype=pl.Int64
            )
        )

        # Use ParetoSelector for automated selection
        selector = ParetoSelector()
        experiments_pl = (
            pl.from_pandas(experiments_df) if not experiments_df.empty else None
        )
        selected_recipes_pl = selector.select_recipes(
            pareto_front=pareto_filtered_pl,
            investigation_space=investigation_space,
            optimization_problem=optimization_problem,
            n_samples=batch_size,
            experiments_df=experiments_pl,
        )

        # Extract original indices from selected recipes
        selected_original_indices = selected_recipes_pl["__original_idx"].to_list()

        # Convert to pinned data format
        result = {str(idx): True for idx in selected_original_indices}
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
