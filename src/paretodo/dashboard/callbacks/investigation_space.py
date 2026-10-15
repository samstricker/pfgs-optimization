"""Callbacks for investigation space visualization."""

__all__ = ["update_investigation_space_plots"]

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output
from plotly.subplots import make_subplots

from paretodo.data_models import InvestigationSpace

from ..server import app
from ..utils.plotting_utils import build_hovertemplate
from ..utils.types_and_constants import (
    LOCAL_TEMP_DIR,
    ClickedPointsData,
    MARKER_SYMBOLS,
    MARKER_COLORS,
    MARKER_NAMES,
    RenamesDict,
)
from ..theme.plotly_theme import get_square_plot_dimensions, get_theme
from ..utils.data_utils import load_manufacturing_df, load_optimization_problem


@app.callback(
    Output("investigation-space-graph", "figure"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    Input("clicked-points-store", "data"),
    Input("pinned-points-store", "data"),
    Input("investigation-space-renames-store", "data"),
    Input("measurement-renames-store", "data"),
    Input("objective-renames-store", "data"),
    Input("objectives-bounds-store", "data"),
    Input("viewport-width-store", "data"),
    Input("pareto-mode-store", "data"),
    prevent_initial_call=True,
)
def update_investigation_space_plots(
    mlflow_run: str | None,
    iteration: str | None,
    clicked_points: ClickedPointsData,
    pinned_points: dict[str, bool] | None,
    investigation_space_renames: dict[str, str],
    measurement_renames: RenamesDict,
    objective_renames: dict[str, str],
    objectives_bounds: dict | None,
    viewport_width: int | None = None,
    pareto_mode: str | None = None,
) -> go.Figure:
    """Create x-y scatter plots for investigation space dimensions using subplots.

    Creates plots showing 2 dimensions at a time, organized in 3 columns.
    Plots pairs: (1,2), (3,4), (5,6), (7,8), etc.
    Marks experiments as green squares and highlights selected points.
    """
    if mlflow_run is None or iteration is None:
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        return fig

    try:
        # Load investigation space to get variable names
        investigation_space = InvestigationSpace.from_json(
            Path(f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/investigation_space.json")
        )

        # Load experiments data
        experiments_csv = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/experiments.csv"
        experiments_df = pd.read_csv(experiments_csv)

        # Load Pareto front data based on selected mode
        if pareto_mode == "Manufacturing":
            pareto_df = load_manufacturing_df(mlflow_run, iteration)
            if pareto_df.empty:
                # Error: Manufacturing mode selected but data not available
                fig = go.Figure()
                fig.update_layout(template=get_theme())
                fig.add_annotation(
                    text=(
                        "Error: Manufacturing data (manufacturing_front_predictions.csv) not "
                        "found for this iteration."
                    ),
                    showarrow=False,
                    font=dict(size=14, color="red"),
                )
                return fig
        else:
            pareto_df = pd.read_csv(
                f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/pareto_front_predictions.csv"
            )

        # Load OptimizationProblem to get objective names map
        optimization_problem_data = load_optimization_problem(
            mlflow_run, pareto_mode or "Optimization"
        )
        objective_names_map = {}
        for idx, axis in enumerate(optimization_problem_data.problem):
            objective_names_map[idx] = axis.var_name

        variables = investigation_space.variables
        num_vars = len(variables)

        # Calculate number of plots needed
        num_plots = (num_vars + 1) // 2  # Round up division

        def _column_count_for_viewport(viewport_w: int | None, max_columns: int) -> int:
            """Pick the subplot column count from the current viewport width."""
            if max_columns <= 1:
                return 1

            if not viewport_w or viewport_w <= 0:
                return min(3, max_columns)

            if viewport_w < 1100:
                desired_columns = 1
            elif viewport_w < 1700:
                desired_columns = 2
            else:
                desired_columns = 3

            return max(1, min(desired_columns, max_columns))

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

        plot_idx = 0

        for i in range(0, num_vars, 2):
            row = plot_idx // cols + 1
            col = plot_idx % cols + 1

            if i + 1 < num_vars:
                # We have a pair
                xaxis_name = variables[i].name
                yaxis_name = variables[i + 1].name

                # Add experiments as green squares
                if (
                    xaxis_name in experiments_df.columns
                    and yaxis_name in experiments_df.columns
                ):
                    # Add row index as customdata for tracking
                    customdata_with_index = []
                    for idx, row_data in experiments_df.iterrows():  # type: ignore[assignment]
                        customdata_with_index.append([idx] + row_data.tolist())  # type: ignore

                    fig.add_trace(
                        go.Scatter(
                            x=experiments_df[xaxis_name],
                            y=experiments_df[yaxis_name],
                            mode="markers",
                            marker=dict(
                                color=MARKER_COLORS["experiment"],
                                size=10,
                                symbol=MARKER_SYMBOLS["experiment"],
                                line=dict(width=2, color="black"),
                            ),
                            name=MARKER_NAMES["experiment"],
                            customdata=customdata_with_index,  # type: ignore
                            hovertemplate=build_hovertemplate(
                                experiments_df,
                                investigation_space_renames=investigation_space_renames,
                                measurement_renames=measurement_renames,
                            ),
                            showlegend=(plot_idx == 0),
                        ),
                        row=row,
                        col=col,
                    )

                    # Add orange crosses for selected experiment points
                    selected_exp_indices = []
                    if (
                        isinstance(clicked_points, dict)
                        and "experiment-results" in clicked_points
                    ):
                        selected_exp_indices = clicked_points["experiment-results"]
                        if not isinstance(selected_exp_indices, list):
                            selected_exp_indices = (
                                list(selected_exp_indices)
                                if selected_exp_indices is not None
                                else []
                            )
                        selected_exp_indices = [
                            x for x in selected_exp_indices if x is not None
                        ]

                    if selected_exp_indices:
                        selected_x = []
                        selected_y = []
                        for idx in selected_exp_indices:
                            if idx < len(experiments_df):
                                selected_x.append(experiments_df[xaxis_name].iloc[idx])
                                selected_y.append(experiments_df[yaxis_name].iloc[idx])

                        if selected_x:
                            fig.add_trace(
                                go.Scatter(
                                    x=selected_x,
                                    y=selected_y,
                                    mode="markers",
                                    marker=dict(
                                        size=14,
                                        symbol=MARKER_SYMBOLS["highlight"],
                                        color=MARKER_COLORS["highlight"],
                                        line=dict(width=2, color="black"),
                                    ),
                                    name=MARKER_NAMES["highlight"],
                                    showlegend=(plot_idx == 0),
                                    hovertemplate="Manual Selection<extra></extra>",
                                ),
                                row=row,
                                col=col,
                            )

                # Add Pareto front points as blue circles (or yellow X if selected)
                if xaxis_name in pareto_df.columns and yaxis_name in pareto_df.columns:
                    # Add row index as customdata for tracking
                    customdata_with_index = []
                    for idx, row_data in pareto_df.iterrows():  # type: ignore[assignment]
                        customdata_with_index.append([idx] + row_data.tolist())  # type: ignore

                    # Determine marker properties based on selection
                    clicked_indices = []
                    if (
                        isinstance(clicked_points, dict)
                        and "investigation-space" in clicked_points
                    ):
                        clicked_indices = clicked_points["investigation-space"]
                        if not isinstance(clicked_indices, list):
                            clicked_indices = (
                                list(clicked_indices)
                                if clicked_indices is not None
                                else []
                            )
                        clicked_indices = [x for x in clicked_indices if x is not None]

                    clicked_indices = [int(x) for x in clicked_indices]

                    # Separate unselected and selected Pareto points
                    unselected_mask = ~pareto_df.index.isin(clicked_indices)
                    selected_mask = pareto_df.index.isin(clicked_indices)

                    # Further separate unselected points by outside bounds status
                    unselected_pareto_df = pareto_df[unselected_mask]
                    if not unselected_pareto_df.empty:
                        # Check which unselected points are outside bounds
                        outside_bounds_indices = []
                        inside_bounds_indices = []

                        for idx in unselected_pareto_df.index:
                            is_outside_bounds = False
                            if objectives_bounds and isinstance(
                                objectives_bounds, dict
                            ):
                                for obj_col, bounds in objectives_bounds.items():
                                    if obj_col in pareto_df.columns:
                                        value = pareto_df.loc[idx, obj_col]
                                        try:
                                            min_val = bounds.get("min")
                                            max_val = bounds.get("max")
                                            min_bound = (
                                                float(min_val)
                                                if min_val is not None
                                                else None
                                            )
                                            max_bound = (
                                                float(max_val)
                                                if max_val is not None
                                                else None
                                            )
                                            if isinstance(value, complex):
                                                continue
                                            value_float = float(value)  # type: ignore
                                        except (ValueError, TypeError):
                                            continue

                                        if (
                                            min_bound is not None
                                            and value_float < min_bound
                                        ):
                                            is_outside_bounds = True
                                            break
                                        if (
                                            max_bound is not None
                                            and value_float > max_bound
                                        ):
                                            is_outside_bounds = True
                                            break

                            if is_outside_bounds:
                                outside_bounds_indices.append(int(idx))  # type: ignore
                            else:
                                inside_bounds_indices.append(int(idx))  # type: ignore

                        # Add inside bounds Pareto points (blue circles)
                        if inside_bounds_indices:
                            inside_pareto_df = pareto_df.loc[inside_bounds_indices]
                            inside_customdata = []
                            for idx, row_data in inside_pareto_df.iterrows():  # type: ignore
                                inside_customdata.append(
                                    [idx] + row_data.tolist()
                                )  # type: ignore

                            fig.add_trace(
                                go.Scatter(
                                    x=inside_pareto_df[xaxis_name],
                                    y=inside_pareto_df[yaxis_name],
                                    mode="markers",
                                    marker=dict(
                                        color=MARKER_COLORS["pareto"],
                                        size=8,
                                        symbol="circle",
                                        line=dict(width=1.5, color="black"),
                                    ),
                                    customdata=inside_customdata,  # type: ignore
                                    hovertemplate=build_hovertemplate(
                                        pareto_df,
                                        objective_names_map=objective_names_map,
                                        objective_renames=objective_renames,
                                        investigation_space_renames=investigation_space_renames,
                                        measurement_renames=measurement_renames,
                                    ),
                                    name=MARKER_NAMES["pareto"],
                                    showlegend=(plot_idx == 0),
                                ),
                                row=row,
                                col=col,
                            )

                        # Add outside bounds Pareto points (black circles)
                        if outside_bounds_indices:
                            outside_pareto_df = pareto_df.loc[outside_bounds_indices]
                            outside_customdata = []
                            for idx, row_data in outside_pareto_df.iterrows():  # type: ignore
                                outside_customdata.append(
                                    [idx] + row_data.tolist()
                                )  # type: ignore

                            fig.add_trace(
                                go.Scatter(
                                    x=outside_pareto_df[xaxis_name],
                                    y=outside_pareto_df[yaxis_name],
                                    mode="markers",
                                    marker=dict(
                                        color=MARKER_COLORS["discarded"],
                                        size=8,
                                        symbol="circle",
                                        line=dict(width=1.5, color="black"),
                                    ),
                                    customdata=outside_customdata,  # type: ignore
                                    hovertemplate=build_hovertemplate(
                                        pareto_df,
                                        objective_names_map=objective_names_map,
                                        objective_renames=objective_renames,
                                        investigation_space_renames=investigation_space_renames,
                                        measurement_renames=measurement_renames,
                                    ),
                                    name=MARKER_NAMES["discarded"],
                                    showlegend=(plot_idx == 0),
                                ),
                                row=row,
                                col=col,
                            )

                    # Add selected Pareto points with separate trace for legend
                    selected_pareto_df = pareto_df[selected_mask]
                    if not selected_pareto_df.empty:
                        selected_customdata = []
                        for idx, row_data in selected_pareto_df.iterrows():  # type: ignore
                            selected_customdata.append([idx] + row_data.tolist())  # type: ignore

                        fig.add_trace(
                            go.Scatter(
                                x=selected_pareto_df[xaxis_name],
                                y=selected_pareto_df[yaxis_name],
                                mode="markers",
                                marker=dict(
                                    color=MARKER_COLORS["selected"],
                                    size=14,
                                    symbol=MARKER_SYMBOLS["selected"],
                                    line=dict(width=1.5, color="black"),
                                ),
                                customdata=selected_customdata,  # type: ignore
                                hovertemplate=build_hovertemplate(
                                    pareto_df,
                                    objective_names_map=objective_names_map,
                                    objective_renames=objective_renames,
                                    investigation_space_renames=investigation_space_renames,
                                    measurement_renames=measurement_renames,
                                ),
                                name=MARKER_NAMES["selected"],
                                showlegend=(plot_idx == 0),
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
                                        pinned_x.append(pareto_df.loc[idx, xaxis_name])
                                        pinned_y.append(pareto_df.loc[idx, yaxis_name])
                                except (ValueError, KeyError):
                                    pass

                        if pinned_x:
                            fig.add_trace(
                                go.Scatter(
                                    x=pinned_x,
                                    y=[float(val) for val in pinned_y],  # type: ignore
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

                # Update axes labels
                display_xaxis_name = (
                    investigation_space_renames.get(xaxis_name, xaxis_name)
                    if investigation_space_renames
                    else xaxis_name
                )
                display_yaxis_name = (
                    investigation_space_renames.get(yaxis_name, yaxis_name)
                    if investigation_space_renames
                    else yaxis_name
                )
                fig.update_xaxes(title_text=display_xaxis_name, row=row, col=col)
                fig.update_yaxes(title_text=display_yaxis_name, row=row, col=col)

            else:
                # Only one variable left, create a 1D scatter plot
                xaxis_name = variables[i].name

                if xaxis_name in experiments_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=experiments_df[xaxis_name],
                            y=[1] * len(experiments_df),
                            mode="markers",
                            marker=dict(
                                color=MARKER_COLORS["experiment"],
                                size=10,
                                symbol="square",
                                line=dict(width=2, color="black"),
                            ),
                            name=MARKER_NAMES["experiment"],
                            customdata=experiments_df[  # type: ignore
                                experiments_df.columns.tolist()
                            ],
                            hovertemplate="<br>".join(
                                [
                                    (
                                        (
                                            f"<b>"
                                            f"{investigation_space_renames.get(col_name, col_name)}"
                                            f"</b>: %{{customdata[{j}]:.4f}}"
                                        )
                                        if experiments_df[col_name].dtype
                                        in ["float64", "int64", "int32", "float32"]
                                        else (
                                            f"<b>"
                                            f"{investigation_space_renames.get(col_name, col_name)}"
                                            f"</b>: %{{customdata[{j}]}}"
                                        )
                                    )
                                    for j, col_name in enumerate(experiments_df.columns)
                                ]
                            )
                            + "<extra></extra>",
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

                    # Add orange crosses for selected experiment points
                    selected_exp_indices = []
                    if (
                        isinstance(clicked_points, dict)
                        and "experiment-results" in clicked_points
                    ):
                        selected_exp_indices = clicked_points["experiment-results"]
                        if not isinstance(selected_exp_indices, list):
                            selected_exp_indices = (
                                list(selected_exp_indices)
                                if selected_exp_indices is not None
                                else []
                            )
                        selected_exp_indices = [
                            x for x in selected_exp_indices if x is not None
                        ]

                    if selected_exp_indices:
                        selected_x = []
                        for idx in selected_exp_indices:
                            if idx < len(experiments_df):
                                selected_x.append(experiments_df[xaxis_name].iloc[idx])

                        if selected_x:
                            fig.add_trace(
                                go.Scatter(
                                    x=selected_x,
                                    y=[1] * len(selected_x),
                                    mode="markers",
                                    marker=dict(
                                        size=14,
                                        symbol=MARKER_SYMBOLS["selected"],
                                        color=MARKER_COLORS["selected"],
                                        line=dict(width=2, color="black"),
                                    ),
                                    showlegend=False,
                                    hoverinfo="skip",
                                ),
                                row=row,
                                col=col,
                            )

                # Update axis labels
                display_xaxis_name = (
                    investigation_space_renames.get(xaxis_name, xaxis_name)
                    if investigation_space_renames
                    else xaxis_name
                )
                fig.update_xaxes(title_text=display_xaxis_name, row=row, col=col)
                fig.update_yaxes(title_text="", row=row, col=col)

            plot_idx += 1

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
    except Exception:  # pylint: disable=broad-exception-caught
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        fig.add_annotation(text="Error creating investigation space plots")
        return fig
