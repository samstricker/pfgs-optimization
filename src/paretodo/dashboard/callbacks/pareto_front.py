"""Callbacks for Pareto front visualization and details table."""

__all__ = [
    "update_pareto_front_plots",
    "update_pareto_details_table",
    "update_pareto_front_parcoords",
    "update_parcoords_column_options",
]

from typing import Any
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, callback_context, html
from plotly.subplots import make_subplots

from ..server import app
from ..utils.table_components import create_pareto_details_table
from ..utils.plotting_utils import build_hovertemplate
from ..utils.data_utils import load_manufacturing_df, load_optimization_problem
from ..theme.plotly_theme import get_square_plot_dimensions
from ..utils.types_and_constants import (
    LOCAL_TEMP_DIR,
    BoundsDict,
    ClickedPointsData,
    MARKER_SYMBOLS,
    MARKER_COLORS,
    MARKER_NAMES,
    RenamesDict,
)
from ..theme.plotly_theme import get_theme


def _extract_name_from_pattern(display_name: str) -> tuple[str | None, str | None]:
    """Extract prefix and inner name from patterns like 'μ(name)' or 'R(name)'.

    Matches patterns like 'μ(name)', 'R(name)', or 'prefix μ(name)', 'prefix R(name)'.
    Returns:
        Tuple of (prefix, inner_name) or (None, None) if pattern doesn't match
    """
    # Match patterns like μ(name), R(name), etc. anywhere in the string
    import re

    match = re.search(r"([μR])\(([^)]+)\)", display_name)
    if match:
        return match.group(1), match.group(2)
    return None, None


def _should_add_diagonal_line(x_display: str, y_display: str) -> bool:
    """Check if x and y axes have same underlying name with different prefixes."""
    x_prefix, x_name = _extract_name_from_pattern(x_display)
    y_prefix, y_name = _extract_name_from_pattern(y_display)

    # Return True if both have patterns, have same inner name, and different prefixes
    if x_name is not None and y_name is not None:
        return x_name == y_name and x_prefix != y_prefix
    return False


def _column_count_for_viewport(viewport_width: int | None, max_columns: int) -> int:
    """Pick the Pareto subplot column count from the current viewport width."""
    if max_columns <= 1:
        return 1

    if not viewport_width or viewport_width <= 0:
        return min(3, max_columns)

    if viewport_width < 1100:
        desired_columns = 1
    elif viewport_width < 1700:
        desired_columns = 2
    else:
        desired_columns = 3

    return max(1, min(desired_columns, max_columns))


@app.callback(
    Output("pareto-front-graph", "figure"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    Input("clicked-points-store", "data"),
    Input("objectives-bounds-store", "data"),
    Input("pinned-points-store", "data"),
    Input("objective-renames-store", "data"),
    Input("investigation-space-renames-store", "data"),
    Input("measurement-renames-store", "data"),
    Input("pareto-mode-store", "data"),
    Input("viewport-width-store", "data"),
    prevent_initial_call=True,
)
def update_pareto_front_plots(
    mlflow_run: str | None,
    iteration: str | None,
    clicked_points: ClickedPointsData,
    objectives_bounds: BoundsDict,
    pinned_points: dict[str, bool],
    objective_renames: dict[str, str],
    investigation_space_renames: dict[str, str],
    measurement_renames: RenamesDict,
    pareto_mode: str,
    viewport_width: int | None = None,
) -> go.Figure:
    """Create x-y scatter plots for all objective combinations using subplots.

    Creates all unique combinations of objectives in a 3-column layout.
    For 3 objectives: (1,2), (1,3), (2,3)
    For 4 objectives: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    Marks Pareto front points as blue circles.
    """
    if mlflow_run is None or iteration is None:
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        return fig

    try:
        # Load Pareto front data based on selected mode
        if pareto_mode == "Manufacturing":
            pareto_df = load_manufacturing_df(mlflow_run, iteration)
            if pareto_df.empty:
                # Error: Manufacturing mode selected but data not available
                fig = go.Figure()
                fig.update_layout(template=get_theme())
                fig.add_annotation(
                    text="Error: Manufacturing data (manufacturing_front_predictions.csv) not found for this iteration. Please check the MLflow artifacts.",
                    showarrow=False,
                    font=dict(size=14, color="red"),
                )
                return fig
        else:
            pareto_csv = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/pareto_front_predictions.csv"
            pareto_df = pd.read_csv(pareto_csv)

        # Load OptimizationProblem to get actual objective names
        optimization_problem_data = load_optimization_problem(mlflow_run, pareto_mode)

        # Create mapping from objective index to objective var_name
        objective_names_map = {}
        for idx, axis in enumerate(optimization_problem_data.problem):
            objective_names_map[idx] = axis.var_name

        # Extract only objective columns (those starting with "obj:")
        objective_columns = [col for col in pareto_df.columns if col.startswith("obj:")]
        num_objectives = len(objective_columns)

        # Generate all unique pairs of objectives without repetition
        objective_pairs = []
        subplot_titles = []
        for i in range(num_objectives):
            for j in range(i + 1, num_objectives):
                objective_pairs.append((i, j))
                obj_x_name = objective_columns[i]
                obj_y_name = objective_columns[j]
                subplot_titles.append(f"{obj_x_name} vs {obj_y_name}")

        if not objective_pairs:
            fig = go.Figure()
            fig.update_layout(template=get_theme())
            fig.add_annotation(text="No objective combinations to display")
            return fig

        # Calculate grid dimensions
        num_plots = len(objective_pairs)
        cols = _column_count_for_viewport(viewport_width, num_plots)
        rows = (num_plots + cols - 1) // cols  # Round up division

        # Create subplot figure
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)],
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
        )
        fig.update_layout(template=get_theme())

        # Add traces for each objective pair
        for plot_idx, (i, j) in enumerate(objective_pairs):
            row = plot_idx // cols + 1
            col = plot_idx % cols + 1

            obj_x_name = objective_columns[i]
            obj_y_name = objective_columns[j]

            if (
                obj_x_name not in pareto_df.columns
                or obj_y_name not in pareto_df.columns
            ):
                continue

            # Add row index as customdata for tracking
            customdata_with_index: list[list[Any]] = []
            for idx, row_data in pareto_df.iterrows():  # type: ignore[assignment]
                customdata_with_index.append(
                    [int(idx)] + row_data.tolist()  # type: ignore[arg-type]
                )

            # Determine marker properties based on selection
            clicked_indices = []
            if (
                clicked_points
                and isinstance(clicked_points, dict)
                and "pareto-front" in clicked_points
            ):
                clicked_indices = clicked_points["pareto-front"]
                if not isinstance(clicked_indices, list):
                    clicked_indices = list(clicked_indices) if clicked_indices else []
                clicked_indices = [int(x) for x in clicked_indices if x is not None]

            # Split pareto points into inside/outside groups so export keeps colors stable
            pareto_inside_bounds = []
            pareto_outside_bounds = []

            for idx in pareto_df.index:

                # Check if point is outside ANY specified bounds
                is_outside_bounds = False
                if objectives_bounds and isinstance(objectives_bounds, dict):
                    for obj_col, bounds in objectives_bounds.items():
                        if obj_col in pareto_df.columns:
                            value = pareto_df.loc[idx, obj_col]
                            # Convert bounds to float if they're strings
                            try:
                                min_val = bounds.get("min")
                                max_val = bounds.get("max")
                                min_bound = (
                                    float(min_val)  # type: ignore
                                    if min_val is not None
                                    else None
                                )
                                max_bound = (
                                    float(max_val)  # type: ignore
                                    if max_val is not None
                                    else None
                                )
                                # Convert value to float for comparison
                                try:
                                    if isinstance(value, complex):
                                        continue
                                    value_float = float(value)  # type: ignore
                                except (ValueError, TypeError):
                                    continue
                            except (ValueError, TypeError):
                                continue

                            if min_bound is not None and value_float < min_bound:
                                is_outside_bounds = True
                                break
                            if max_bound is not None and value_float > max_bound:
                                is_outside_bounds = True
                                break

                if is_outside_bounds:
                    pareto_outside_bounds.append(idx)
                else:
                    pareto_inside_bounds.append(idx)

            # Build hovertemplate with renamed display names
            hovertemplate = build_hovertemplate(
                pareto_df,
                objective_names_map=objective_names_map,
                objective_renames=objective_renames,
                investigation_space_renames=investigation_space_renames,
                measurement_renames=measurement_renames,
            )

            if pareto_inside_bounds:
                inside_pareto_df = pareto_df.loc[pareto_inside_bounds]
                inside_customdata = []
                for idx, row_data in inside_pareto_df.iterrows():  # type: ignore[assignment]
                    inside_customdata.append([int(idx)] + row_data.tolist())  # type: ignore[arg-type]

                fig.add_trace(
                    go.Scatter(
                        x=inside_pareto_df[obj_x_name],
                        y=inside_pareto_df[obj_y_name],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=MARKER_COLORS["pareto"],
                            line=dict(width=2, color="black"),
                            symbol="circle",
                        ),
                        name=MARKER_NAMES["pareto"],
                        customdata=inside_customdata,  # type: ignore
                        hovertemplate=hovertemplate,
                        showlegend=(plot_idx == 0),
                    ),
                    row=row,
                    col=col,
                )

            if pareto_outside_bounds:
                outside_pareto_df = pareto_df.loc[pareto_outside_bounds]
                outside_customdata = []
                for idx, row_data in outside_pareto_df.iterrows():  # type: ignore[assignment]
                    outside_customdata.append([int(idx)] + row_data.tolist())  # type: ignore[arg-type]

                fig.add_trace(
                    go.Scatter(
                        x=outside_pareto_df[obj_x_name],
                        y=outside_pareto_df[obj_y_name],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=MARKER_COLORS["discarded"],
                            line=dict(width=2, color="black"),
                            symbol="circle",
                        ),
                        name=MARKER_NAMES["discarded"],
                        customdata=outside_customdata,  # type: ignore
                        hovertemplate=hovertemplate,
                        showlegend=(plot_idx == 0),
                    ),
                    row=row,
                    col=col,
                )

            # Add manual selected points as orange crosses
            if clicked_indices:
                manual_x = []
                manual_y = []
                for idx in clicked_indices:
                    if idx in pareto_df.index:
                        manual_x.append(pareto_df.loc[idx, obj_x_name])
                        manual_y.append(pareto_df.loc[idx, obj_y_name])

                if manual_x:
                    fig.add_trace(
                        go.Scatter(
                            x=manual_x,
                            y=[float(v) for v in manual_y],  # type: ignore
                            mode="markers",
                            marker=dict(
                                color=MARKER_COLORS["selected"],
                                size=14,
                                symbol=MARKER_SYMBOLS["selected"],
                                line=dict(width=2, color="black"),
                            ),
                            name=MARKER_NAMES["selected"],
                            showlegend=(plot_idx == 0),
                            hovertemplate="Manual Selection<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )

            # Add pinned points as black crosses
            if pinned_points and isinstance(pinned_points, dict):
                pinned_x = []
                pinned_y = []
                for idx_str in pinned_points.keys():
                    if pinned_points[idx_str]:
                        try:
                            idx = int(idx_str)
                            if idx in pareto_df.index:
                                pinned_x.append(pareto_df.loc[idx, obj_x_name])
                                pinned_y.append(pareto_df.loc[idx, obj_y_name])
                        except (ValueError, KeyError):
                            pass

                if pinned_x:
                    fig.add_trace(
                        go.Scatter(
                            x=pinned_x,
                            y=[float(v) for v in pinned_y],  # type: ignore
                            mode="markers",
                            marker=dict(
                                color=MARKER_COLORS["auto_selected"],
                                size=12,
                                symbol=MARKER_SYMBOLS["auto_selected"],
                                line=dict(width=2, color="black"),
                            ),
                            name=MARKER_NAMES["auto_selected"],
                            showlegend=(plot_idx == 0),
                            hovertemplate="Auto Selection<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )

            # Add shaded regions based on bounds
            if objectives_bounds and isinstance(objectives_bounds, dict):
                if obj_x_name in objectives_bounds:
                    bounds_x = objectives_bounds[obj_x_name]
                    min_x = bounds_x.get("min")
                    max_x = bounds_x.get("max")
                    if min_x is not None:
                        fig.add_vline(
                            x=float(min_x),  # type: ignore
                            line=dict(color="red", width=2, dash="dash"),
                            row=row,
                            col=col,
                        )
                    if max_x is not None:
                        fig.add_vline(
                            x=float(max_x),  # type: ignore
                            line=dict(color="red", width=2, dash="dash"),
                            row=row,
                            col=col,
                        )

                if obj_y_name in objectives_bounds:
                    bounds_y = objectives_bounds[obj_y_name]
                    min_y = bounds_y.get("min")
                    max_y = bounds_y.get("max")
                    if min_y is not None:
                        fig.add_hline(
                            y=float(min_y),  # type: ignore
                            line=dict(color="red", width=2, dash="dash"),
                            row=row,
                            col=col,
                        )
                    if max_y is not None:
                        fig.add_hline(
                            y=float(max_y),  # type: ignore
                            line=dict(color="red", width=2, dash="dash"),
                            row=row,
                            col=col,
                        )

            # Update axes labels with renames
            obj_x_display = objective_names_map.get(i, obj_x_name.replace("obj:", ""))
            obj_y_display = objective_names_map.get(j, obj_y_name.replace("obj:", ""))

            if objective_renames and isinstance(objective_renames, dict):
                obj_x_display = objective_renames.get(str(i), obj_x_display)
                obj_y_display = objective_renames.get(str(j), obj_y_display)

            fig.update_xaxes(title_text=obj_x_display, row=row, col=col)
            fig.update_yaxes(title_text=obj_y_display, row=row, col=col)

            # Add diagonal X=Y line if x and y represent same variable with different prefixes
            if _should_add_diagonal_line(obj_x_display, obj_y_display):
                # Calculate range for diagonal line
                try:
                    x_min = pareto_df[obj_x_name].min()
                    x_max = pareto_df[obj_x_name].max()
                    y_min = pareto_df[obj_y_name].min()
                    y_max = pareto_df[obj_y_name].max()

                    # Diagonal line range should be the min/max of both axes
                    min_val = min(float(x_min), float(y_min))  # type: ignore
                    max_val = max(float(x_max), float(y_max))  # type: ignore

                    # Add diagonal line as a shape
                    fig.add_shape(
                        type="line",
                        x0=min_val,
                        y0=min_val,
                        x1=max_val,
                        y1=max_val,
                        line=dict(color="gray", width=2, dash="dot"),
                        row=row,
                        col=col,
                    )
                except (ValueError, TypeError):
                    pass  # Skip if calculation fails

        width, height = get_square_plot_dimensions(
            columns=cols,
            rows=rows,
            horizontal_spacing=0.12,
            vertical_spacing=0.15,
        )
        fig.update_layout(
            width=width,
            height=height,
            showlegend=True,
            hovermode="closest",
        )

        return fig

    except FileNotFoundError as e:
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        fig.add_annotation(text=f"Data not found: {e}")
        return fig
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"ERROR in update_pareto_front_plots: {e}")
        print(traceback.format_exc())
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        fig.add_annotation(text=f"Error: {str(e)}")
        return fig


@app.callback(
    Output("pareto-details-table-container", "children"),
    Output("pinned-points-store", "data", allow_duplicate=True),
    Input("clicked-points-store", "data"),
    Input("pinned-points-store", "data"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    Input({"type": "pin-checkbox", "index": ALL}, "value"),
    Input("objective-renames-store", "data"),
    Input("investigation-space-renames-store", "data"),
    Input("measurement-renames-store", "data"),
    Input("pareto-mode-store", "data"),
    prevent_initial_call=True,
)
def update_pareto_details_table(
    clicked_points: ClickedPointsData,
    pinned_data: dict[str, bool],
    mlflow_run: str | None,
    iteration: str | None,
    pin_values: list[Any],
    objective_renames: dict[str, str],
    investigation_space_renames: dict[str, str],
    measurement_renames: RenamesDict,
    pareto_mode: str,
) -> tuple[html.Div, dict[str, bool]]:
    """Create a table showing detailed values for selected and pinned Pareto points."""
    if mlflow_run is None or iteration is None:
        return html.Div("No data selected"), pinned_data

    try:
        # Load Pareto front data based on selected mode
        if pareto_mode == "Manufacturing":
            pareto_df = load_manufacturing_df(mlflow_run, iteration)
            if pareto_df.empty:
                # Error: Manufacturing mode selected but data not available
                return (
                    html.Div(
                        "Error: Manufacturing data (manufacturing_front_predictions.csv) not found for this iteration.",
                        style={"color": "red", "fontWeight": "bold", "padding": "10px"},
                    ),
                    pinned_data,
                )
        else:
            pareto_df = pd.read_csv(
                f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/pareto_front_predictions.csv"
            )

        # Load OptimizationProblem
        optimization_problem_data = load_optimization_problem(mlflow_run, pareto_mode)

        objective_names_map = {}
        for idx, axis in enumerate(optimization_problem_data.problem):
            objective_names_map[idx] = axis.var_name

        # Get all clicked points
        clicked_indices = []
        if isinstance(clicked_points, dict) and "pareto-front" in clicked_points:
            indices = clicked_points["pareto-front"]
            if isinstance(indices, list):
                clicked_indices = [int(x) for x in indices if x is not None]

        # Check what triggered callback
        ctx = callback_context
        trigger_id = (
            ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
        )

        # Update pinned points based on checkbox changes
        if trigger_id and trigger_id.startswith('{"type":"pin-checkbox"'):
            new_pinned_data = {}
            if pin_values:
                for pin_value in pin_values:
                    if pin_value and len(pin_value) > 0:
                        new_pinned_data[str(pin_value[0])] = True
        else:
            new_pinned_data = pinned_data if pinned_data else {}

        # Get points to display with selection type
        points_to_show = {}
        # Add manually selected points with "Manual" label
        for idx in clicked_indices:
            points_to_show[f"{MARKER_NAMES['selected']} (#{idx})"] = (
                idx,
                MARKER_NAMES["selected"],
            )

        # Add auto-pinned points with "Auto" label
        for idx_str in new_pinned_data.keys():
            if new_pinned_data[idx_str]:
                idx_int = int(idx_str)
                points_to_show[f"{MARKER_NAMES['auto_selected']} (#{idx_int})"] = (
                    idx_int,
                    MARKER_NAMES["auto_selected"],
                )

        if not points_to_show:
            return html.Div("Click on a point to view details"), new_pinned_data

        # Create table using utility function
        table = create_pareto_details_table(
            pareto_df,
            points_to_show,
            objective_names_map,
            objective_renames,
            investigation_space_renames,
            measurement_renames,
        )

        return table, new_pinned_data

    except Exception as e:  # pylint: disable=broad-exception-caught
        return html.Div(f"Error loading pareto details: {str(e)}"), pinned_data


@app.callback(
    Output("parcoords-columns-checklist", "options"),
    Output("parcoords-columns-checklist", "value"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    Input("pareto-mode-store", "data"),
    prevent_initial_call=True,
)
def update_parcoords_column_options(
    mlflow_run: str | None,
    iteration: str | None,
    pareto_mode: str,
) -> tuple[list[dict], list[str]]:
    """Populate column selection checklist with available columns."""
    if mlflow_run is None or iteration is None:
        return [], []

    try:
        # Load Pareto front data based on selected mode
        if pareto_mode == "Manufacturing":
            pareto_df = load_manufacturing_df(mlflow_run, iteration)
            if pareto_df.empty:
                # Error: Manufacturing mode selected but data not available
                print("Error: Manufacturing data not found for parallel coordinates")
                return [], []
        else:
            pareto_csv = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/pareto_front_predictions.csv"
            pareto_df = pd.read_csv(pareto_csv)

        # Get numeric columns
        numeric_cols = pareto_df.select_dtypes(include=[np.number]).columns.tolist()

        # Create options for checklist
        options = [{"label": col, "value": col} for col in numeric_cols]

        # All columns selected by default
        return options, numeric_cols

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error loading parcoords column options: {e}")
        return [], []


@app.callback(
    Output("pareto-front-parcoords-graph", "figure"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    Input("clicked-points-store", "data"),
    Input("pinned-points-store", "data"),
    Input("objective-renames-store", "data"),
    Input("investigation-space-renames-store", "data"),
    Input("measurement-renames-store", "data"),
    Input("parcoords-columns-checklist", "value"),
    Input("parcoords-visibility-checklist", "value"),
    Input("pareto-mode-store", "data"),
    prevent_initial_call=True,
)
def update_pareto_front_parcoords(
    mlflow_run: str | None,
    iteration: str | None,
    clicked_points: ClickedPointsData,
    pinned_points: dict[str, bool],
    objective_renames: dict[str, str],
    investigation_space_renames: dict[str, str],
    measurement_renames: RenamesDict,
    selected_columns: list[str],
    visibility_options: list[str],
    pareto_mode: str,
) -> go.Figure:
    """Create a parallel coordinates plot for Pareto front showing all columns."""
    if mlflow_run is None or iteration is None:
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        return fig

    try:
        # Load Pareto front data based on selected mode
        if pareto_mode == "Manufacturing":
            pareto_df = load_manufacturing_df(mlflow_run, iteration)
            if pareto_df.empty:
                # Error: Manufacturing mode selected but data not available
                fig = go.Figure()
                fig.update_layout(template=get_theme())
                fig.add_annotation(
                    text="Error: Manufacturing data (manufacturing_front_predictions.csv) not found for this iteration.",
                    showarrow=False,
                    font=dict(size=14, color="red"),
                )
                return fig
        else:
            pareto_csv = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/pareto_front_predictions.csv"
            pareto_df = pd.read_csv(pareto_csv)

        # Load experiments data
        try:
            experiments_csv = (
                f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/experiments.csv"
            )
            experiments_df = pd.read_csv(experiments_csv)
        except (FileNotFoundError, OSError):
            experiments_df = None

        # Load OptimizationProblem to get actual objective names
        optimization_problem_data = load_optimization_problem(mlflow_run, pareto_mode)

        # Create mapping from objective index to objective var_name
        objective_names_map = {}
        for idx, axis in enumerate(optimization_problem_data.problem):
            objective_names_map[idx] = axis.var_name

        # Combine pareto and experiments data
        if experiments_df is not None:
            # Get all columns from both dataframes
            all_cols = set(pareto_df.columns) | set(experiments_df.columns)

            # Ensure all columns exist in both dataframes, filling with NaN where missing
            for col in all_cols:
                if col not in pareto_df.columns:
                    pareto_df[col] = np.nan
                if col not in experiments_df.columns:
                    experiments_df[col] = np.nan

        # Get selected/clicked indices (only for pareto front)
        clicked_indices = []
        if (
            clicked_points
            and isinstance(clicked_points, dict)
            and "pareto-front" in clicked_points
        ):
            clicked_indices = clicked_points["pareto-front"]
            if not isinstance(clicked_indices, list):
                clicked_indices = list(clicked_indices) if clicked_indices else []
            clicked_indices = [int(x) for x in clicked_indices if x is not None]

        # Get pinned indices (only for pareto front)
        pinned_indices = []
        if pinned_points and isinstance(pinned_points, dict):
            pinned_indices = [int(k) for k, v in pinned_points.items() if v]

        # Parse visibility options
        show_experiments = "show_experiments" in (visibility_options or [])
        show_selected = "show_selected" in (visibility_options or [])

        # Create a dataframe for the plot based on visibility options
        plot_df = pd.DataFrame()

        # Add pareto data based on visibility
        pareto_to_add = pareto_df.copy()
        if not show_selected and experiments_df is not None:
            # Filter out selected/pinned points from pareto
            selected_mask = pareto_to_add.index.isin(
                clicked_indices
            ) | pareto_to_add.index.isin(pinned_indices)
            pareto_to_add = pareto_to_add[~selected_mask].copy()

        plot_df = pd.concat([plot_df, pareto_to_add], ignore_index=True)

        # Add experiments if enabled
        if show_experiments and experiments_df is not None:
            plot_df = pd.concat([plot_df, experiments_df], ignore_index=True)

        # Select numeric columns from plot data
        numeric_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter to only selected columns (from user checklist)
        if selected_columns:
            numeric_cols = [col for col in numeric_cols if col in selected_columns]

        if not numeric_cols:
            fig = go.Figure()
            fig.update_layout(template=get_theme())
            fig.add_annotation(text="No numeric columns to display")
            return fig

        # Create dimensions for parallel coordinates plot
        dimensions = []
        for col in numeric_cols:
            # Determine the display name based on column type
            label = col

            # For objectives (obj:X columns)
            if col.startswith("obj:"):
                objective_columns = [c for c in numeric_cols if c.startswith("obj:")]
                col_idx = objective_columns.index(col)
                obj_display = objective_names_map.get(col_idx, col.replace("obj:", ""))
                if objective_renames and isinstance(objective_renames, dict):
                    obj_display = objective_renames.get(str(col_idx), obj_display)
                label = obj_display
            # For investigation space variables
            elif investigation_space_renames and isinstance(
                investigation_space_renames, dict
            ):
                label = investigation_space_renames.get(col, col)
            # For measurements
            elif measurement_renames and isinstance(measurement_renames, dict):
                label = measurement_renames.get(col, col)

            try:
                min_val = plot_df[col].min()
                max_val = plot_df[col].max()
                dimensions.append(
                    dict(
                        label=label,
                        values=plot_df[
                            col
                        ].tolist(),  # Convert to list; NaN values are preserved
                        range=[float(min_val), float(max_val)],
                    )
                )
            except (ValueError, TypeError):
                # Skip columns that can't be converted to numeric
                continue

        if not dimensions:
            fig = go.Figure()
            fig.update_layout(template=get_theme())
            fig.add_annotation(text="No valid numeric columns to display")
            return fig

        # Create numeric color array for selected vs non-selected vs experiments
        # 0 = pareto non-selected (blue)
        # 1 = pareto selected/pinned (orange)
        # 2 = experiments (green)
        colors = []

        # Add colors for pareto points that are being displayed
        if show_selected:
            # Show all pareto points
            for idx in range(len(pareto_df)):
                if idx in pinned_indices or idx in clicked_indices:
                    colors.append(1)
                else:
                    colors.append(0)
        else:
            # Only show non-selected pareto points
            for idx in range(len(pareto_df)):
                if not (idx in pinned_indices or idx in clicked_indices):
                    colors.append(0)

        # Add colors for experiments if enabled
        if show_experiments and experiments_df is not None:
            colors.extend([2] * len(experiments_df))

        # Determine cmin/cmax based on what colors are actually present
        if not colors:
            # No data to display
            fig = go.Figure()
            fig.update_layout(template=get_theme())
            fig.add_annotation(text="No data to display with current filter settings")
            return fig

        # Create parallel coordinates plot with custom discrete colorscale
        # Always use cmin=0, cmax=2 to keep color mapping consistent regardless of visibility toggles
        fig = go.Figure(
            data=go.Parcoords(
                dimensions=dimensions,
                line=dict(
                    color=colors,
                    cmin=0,
                    cmax=2,
                    colorscale=[
                        [0.0, MARKER_COLORS["pareto"]],  # value 0 (norm 0.0): blue
                        [0.32, MARKER_COLORS["pareto"]],  # value 0 (norm 0.32): blue
                        [
                            0.33,
                            MARKER_COLORS["selected"],
                        ],  # value 1 (norm 0.33): orange
                        [
                            0.66,
                            MARKER_COLORS["selected"],
                        ],  # value 1 (norm 0.66): orange
                        [0.67, "#00FF00"],  # value 2 (norm 0.67): bright green
                        [1.0, "#00FF00"],  # value 2 (norm 1.0): bright green
                    ],
                    showscale=False,
                ),
                labelangle=-45,
                labelside="bottom",
            )
        )

        fig.update_layout(
            template=get_theme(),
            height=700,
            title="Pareto Front - Parallel Coordinates",
            font=dict(size=12),
            margin=dict(l=80, r=40, b=150),
        )

        return fig

    except FileNotFoundError as e:
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        fig.add_annotation(text=f"Data not found: {e}")
        return fig
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error creating pareto front parallel coordinates: {e}")
        traceback.print_exc()
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        fig.add_annotation(text=f"Error: {e}")
        return fig
