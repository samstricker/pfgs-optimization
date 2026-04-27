"""Callbacks for Pareto front visualization and details table."""

__all__ = [
    "update_pareto_front_plots",
    "update_pareto_details_table",
]

from pathlib import Path
from typing import Any
import traceback

import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, callback_context, html
from plotly.subplots import make_subplots

from paretodo.data_models import OptimizationProblem

from ..server import app
from ..utils.table_components import create_pareto_details_table
from ..utils.plotting_utils import build_hovertemplate
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
        # Load Pareto front data
        pareto_csv = (
            f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/pareto_front_predictions.csv"
        )
        pareto_df = pd.read_csv(pareto_csv)

        # Load OptimizationProblem to get actual objective names
        opt_prob_path = Path(
            f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/optimization_problem.json"
        )
        optimization_problem_data = OptimizationProblem.from_json(opt_prob_path)

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
        cols = 3
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

            # Create marker property arrays (excluding clicked points that will be shown separately)
            pareto_indices = pareto_df.index.tolist()
            colors = []
            sizes = []
            symbols = []
            opacity = []

            for idx in pareto_indices:

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
                    colors.append(MARKER_COLORS["discarded"])
                    symbols.append("circle")
                    sizes.append(10)
                    opacity.append(1.0)
                else:
                    colors.append(MARKER_COLORS["pareto"])
                    symbols.append("circle")
                    sizes.append(10)
                    opacity.append(1.0)

            # Build hovertemplate with renamed display names
            hovertemplate = build_hovertemplate(
                pareto_df,
                objective_names_map=objective_names_map,
                objective_renames=objective_renames,
                investigation_space_renames=investigation_space_renames,
                measurement_renames=measurement_renames,
            )

            fig.add_trace(
                go.Scatter(
                    x=pareto_df[obj_x_name],
                    y=pareto_df[obj_y_name],
                    mode="markers",
                    marker=dict(
                        color=colors,
                        size=sizes,
                        line=dict(width=2, color="black"),
                        symbol=symbols,
                        opacity=opacity,
                    ),
                    name=MARKER_NAMES["pareto"],
                    customdata=customdata_with_index,  # type: ignore
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

        fig.update_layout(
            height=400 * rows + 100,
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
) -> tuple[html.Div, dict[str, bool]]:
    """Create a table showing detailed values for selected and pinned Pareto points."""
    if mlflow_run is None or iteration is None:
        return html.Div("No data selected"), pinned_data

    try:
        # Load Pareto front data
        pareto_df = pd.read_csv(
            f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/pareto_front_predictions.csv"
        )

        # Load OptimizationProblem
        optimization_problem_data = OptimizationProblem.from_json(
            Path(f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/optimization_problem.json")
        )

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
