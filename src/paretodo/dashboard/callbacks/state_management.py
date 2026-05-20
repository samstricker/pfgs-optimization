"""Callbacks for managing renames and objective bounds."""

from dash import Input, Output, State, ALL, callback_context, html
from ..server import app
from ..utils.data_utils import load_pareto_df
from ..utils.table_components import create_objectives_bounds_table
from ..utils.types_and_constants import ObjectiveBoundsData, RenamesDict

__all__ = [
    "update_objectives_table",
    "update_objectives_bounds",
    "update_objective_renames",
    "update_investigation_space_renames",
    "update_measurement_renames",
    "update_alternative_expected_errors",
]


@app.callback(
    Output("objectives-table-container", "children"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    Input("objective-renames-store", "data"),
)
def update_objectives_table(
    mlflow_run: str, iteration: str, objective_renames: RenamesDict
) -> html.Div:
    """Create editable table for objective bounds."""
    if mlflow_run is None:
        return html.Div("No data selected")

    try:
        pareto_df = load_pareto_df(mlflow_run, iteration)
        objective_columns = [col for col in pareto_df.columns if col.startswith("obj:")]

        if not objective_columns:
            return html.Div("No objectives found")

        return create_objectives_bounds_table(objective_columns, objective_renames)

    except FileNotFoundError as e:
        return html.Div(f"Data not found: {e}")
    except (KeyError, ValueError, TypeError) as e:
        return html.Div(f"Error loading objectives: {e}")


@app.callback(
    Output("objectives-bounds-store", "data"),
    Input({"type": "objective-min-input", "index": ALL}, "value"),
    Input({"type": "objective-max-input", "index": ALL}, "value"),
    State({"type": "objective-min-input", "index": ALL}, "id"),
)
def update_objectives_bounds(
    min_values: list[float | None],
    max_values: list[float | None],
    input_ids: list[dict],
) -> ObjectiveBoundsData:
    """Store edited objective bounds."""
    bounds_data: ObjectiveBoundsData = {}

    seen = set()
    unique_indices = []
    for i, input_id in enumerate(input_ids):
        obj_name = input_id["index"]
        if obj_name not in seen:
            seen.add(obj_name)
            unique_indices.append(i)

    for i in unique_indices:
        obj_name = input_ids[i]["index"]
        min_val = min_values[i] if i < len(min_values) else None
        max_val = max_values[i] if i < len(max_values) else None

        # Convert to float, handling None and string values from dcc.Store serialization
        try:
            min_val = float(min_val) if min_val is not None else None
        except (ValueError, TypeError):
            min_val = None

        try:
            max_val = float(max_val) if max_val is not None else None
        except (ValueError, TypeError):
            max_val = None

        bounds_data[obj_name] = {"min": min_val, "max": max_val}

    return bounds_data


@app.callback(
    Output("objective-renames-store", "data"),
    Input({"type": "objective-rename-input", "index": ALL}, "value"),
    State({"type": "objective-rename-input", "index": ALL}, "id"),
    Input("mlflow-run", "value"),
)
def update_objective_renames(
    rename_values: list[str],
    rename_ids: list[dict],
    _mlflow_run: str,
) -> RenamesDict:
    """Store objective renames or reset on new run."""
    ctx = callback_context

    if ctx.triggered and any("mlflow-run" in str(t["prop_id"]) for t in ctx.triggered):
        return {}

    renames_data: RenamesDict = {}
    for i, rename_id in enumerate(rename_ids):
        obj_name = rename_id["index"]
        rename_val = rename_values[i] if i < len(rename_values) else ""

        if rename_val and rename_val.strip():
            renames_data[obj_name] = rename_val.strip()

    return renames_data


@app.callback(
    Output("investigation-space-renames-store", "data"),
    Input({"type": "investigation-space-rename-input", "index": ALL}, "value"),
    State({"type": "investigation-space-rename-input", "index": ALL}, "id"),
    Input("mlflow-run", "value"),
)
def update_investigation_space_renames(
    rename_values: list[str],
    rename_ids: list[dict],
    _mlflow_run: str,
) -> RenamesDict:
    """Store investigation space variable renames or reset on new run."""
    ctx = callback_context

    if ctx.triggered and any("mlflow-run" in str(t["prop_id"]) for t in ctx.triggered):
        return {}

    renames_data: RenamesDict = {}
    for i, rename_id in enumerate(rename_ids):
        var_name = rename_id["index"]
        rename_val = rename_values[i] if i < len(rename_values) else ""

        if rename_val and rename_val.strip():
            renames_data[var_name] = rename_val.strip()

    return renames_data


@app.callback(
    Output("measurement-renames-store", "data"),
    Input({"type": "measurement-rename-input", "index": ALL}, "value"),
    State({"type": "measurement-rename-input", "index": ALL}, "id"),
    Input("mlflow-run", "value"),
)
def update_measurement_renames(
    rename_values: list[str],
    rename_ids: list[dict],
    _mlflow_run: str,
) -> RenamesDict:
    """Store measurement renames or reset on new run."""
    ctx = callback_context

    if ctx.triggered and any("mlflow-run" in str(t["prop_id"]) for t in ctx.triggered):
        return {}

    renames_data: RenamesDict = {}
    for i, rename_id in enumerate(rename_ids):
        measurement_name = rename_id["index"]
        rename_val = rename_values[i] if i < len(rename_values) else ""

        if rename_val and rename_val.strip():
            renames_data[measurement_name] = rename_val.strip()

    return renames_data


@app.callback(
    Output("alternative-expected-error-store", "data"),
    Input({"type": "alternative-expected-error-input", "index": ALL}, "value"),
    State({"type": "alternative-expected-error-input", "index": ALL}, "id"),
    Input("mlflow-run", "value"),
)
def update_alternative_expected_errors(
    error_values: list[str],
    error_ids: list[dict],
    _mlflow_run: str,
) -> dict:
    """Store alternative expected error values or reset on new run."""
    ctx = callback_context

    if ctx.triggered and any("mlflow-run" in str(t["prop_id"]) for t in ctx.triggered):
        return {}

    errors_data = {}
    for i, error_id in enumerate(error_ids):
        var_name = error_id["index"]
        error_val = error_values[i] if i < len(error_values) else ""

        if error_val and error_val.strip():
            try:
                errors_data[var_name] = float(error_val.strip())
            except ValueError:
                pass  # Ignore invalid numeric values

    return errors_data
