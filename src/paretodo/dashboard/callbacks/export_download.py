"""Callbacks for downloading and exporting plots/data."""

# pylint: disable=too-many-lines

from io import BytesIO
import os
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, State, dcc, no_update
from plotly.subplots import make_subplots

from paretodo.data_models import InvestigationSpace

from ..server import app
from ..utils.types_and_constants import (
    LOCAL_TEMP_DIR,
    MARKER_COLORS,
    MARKER_NAMES,
    MARKER_SYMBOLS,
)
from ..utils.plotting_utils import get_display_name
from ..utils.data_utils import (
    load_investigation_space,
    load_experiments_df,
    get_measurement_columns,
    get_variable_pairs,
    load_manufacturing_df,
    load_optimization_problem,
)
from ..theme.plotly_theme import (
    GP_PREDICTION_RIGHT_MARGIN,
    SQUARE_PLOT_RIGHT_MARGIN,
    get_square_plot_dimensions,
    get_theme,
)
from .investigation_space import update_investigation_space_plots
from .pareto_front import update_pareto_front_plots, update_pareto_front_parcoords
from .experiment_results import update_experiment_results_parcoords

__all__ = ["download_current_plots"]


def _slugify(value: str) -> str:
    """Convert a figure title into a file-friendly name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "figure"


def _figure_title(fig: go.Figure) -> str:
    """Return the most useful title text for a Plotly figure."""
    title = fig.layout.title.text if fig.layout.title else None
    return str(title).strip() if title else ""


def _subplot_domains(fig: go.Figure, axis_prefix: str) -> int:
    """Count unique subplot domains along a given axis direction."""
    layout = fig.to_plotly_json().get("layout", {})
    domains: list[tuple[float, float]] = []

    for key, axis in layout.items():
        if not key.startswith(axis_prefix) or not isinstance(axis, dict):
            continue
        domain = axis.get("domain")
        if isinstance(domain, list) and len(domain) == 2:
            domain_pair = (float(domain[0]), float(domain[1]))
            if domain_pair not in domains:
                domains.append(domain_pair)

    return max(len(domains), 1)


def _export_dimensions(fig: go.Figure, plot_type: str = "default") -> tuple[int, int]:
    """Compute consistent export dimensions for a figure based on type."""
    layout = fig.to_plotly_json().get("layout", {})
    title = _figure_title(fig)
    columns = _subplot_domains(fig, "xaxis")
    rows = _subplot_domains(fig, "yaxis")

    # For scatter and contour plots, make each subplot square
    if plot_type in [
        "experiment-results",
        "gp-predictions",
        "pareto-front",
        "investigation-space",
    ]:
        if plot_type == "gp-predictions":
            right_margin = GP_PREDICTION_RIGHT_MARGIN
            horizontal_spacing = 0.12
            vertical_spacing = 0.0
        elif plot_type == "experiment-results":
            right_margin = SQUARE_PLOT_RIGHT_MARGIN
            horizontal_spacing = 0.1
            vertical_spacing = 0.0
        else:
            right_margin = SQUARE_PLOT_RIGHT_MARGIN
            horizontal_spacing = 0.12
            vertical_spacing = 0.15

        width, height = get_square_plot_dimensions(
            columns=columns,
            rows=rows,
            right_margin=right_margin,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
        )
        return width, height

    # Original logic for parallel coordinates and other plots
    height = int(layout.get("height") or 0)
    if height <= 0:
        if "pareto front" in title.lower():
            height = 400 * rows + 100
        else:
            height = 500

    if "pareto front" in title.lower():
        width = 420 * max(columns, 3)
    else:
        width = max(900, 420 * columns)

    return width, height


def _figure_from_store(figure_data: dict[str, Any]) -> go.Figure:
    """Convert serialized Plotly figure data back into a Figure."""
    return go.Figure(figure_data)


def _regenerate_experiment_results_figures(
    mlflow_run: str,
    iteration: str,
    clicked_points: dict[str, list[int]],
    measurement_renames: dict[str, str] | None = None,
    investigation_space_renames: dict[str, str] | None = None,
) -> list[go.Figure]:
    """Regenerate experiment result figures with current state and renames."""
    try:
        investigation_space = load_investigation_space(mlflow_run)
        experiments_df = load_experiments_df(mlflow_run, iteration)
        variables = investigation_space.variables

        if len(variables) < 2:
            return []

        var_names = [v.name for v in variables]
        measurement_cols = get_measurement_columns(experiments_df, var_names)
        var_pairs = get_variable_pairs(variables)

        if not measurement_cols or not var_pairs:
            return []

        selected_indices = (
            clicked_points.get("experiment-results", [])
            if isinstance(clicked_points, dict)
            else []
        )

        figures = []
        num_cols = len(var_pairs)

        for measurement_col in measurement_cols:
            fig = make_subplots(
                rows=1,
                cols=num_cols,
                specs=[[{"secondary_y": False} for _ in range(num_cols)]],
                vertical_spacing=0.12,
                horizontal_spacing=0.1,
            )
            fig.update_layout(template=get_theme())

            for col_idx, (var_x, var_y) in enumerate(var_pairs):
                col = col_idx + 1
                if (
                    var_x not in experiments_df.columns
                    or var_y not in experiments_df.columns
                ):
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=experiments_df[var_x],
                        y=experiments_df[var_y],
                        mode="markers",
                        marker=dict(
                            size=8,
                            symbol="square",
                            color=experiments_df[measurement_col],
                            # colorscale="Viridis",
                            showscale=(col_idx == 0),
                            line=dict(width=0.5, color="DarkSlateGrey"),
                        ),
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )

                selected_x, selected_y = [], []
                for idx in selected_indices:
                    if isinstance(idx, int) and 0 <= idx < len(experiments_df):
                        selected_x.append(experiments_df[var_x].iloc[idx])
                        selected_y.append(experiments_df[var_y].iloc[idx])

                if selected_x:
                    fig.add_trace(
                        go.Scatter(
                            x=selected_x,
                            y=selected_y,
                            mode="markers",
                            marker=dict(
                                size=12,
                                symbol="cross",
                                color="#FF001E",
                                line=dict(width=2, color="black"),
                            ),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=col,
                    )

                x_display = get_display_name(var_x, investigation_space_renames)
                y_display = get_display_name(var_y, investigation_space_renames)
                fig.update_xaxes(title_text=x_display, row=1, col=col)
                fig.update_yaxes(title_text=y_display, row=1, col=col)

            measurement_display = get_display_name(measurement_col, measurement_renames)
            fig.update_layout(
                height=500,
                showlegend=False,
                hovermode="closest",
                title_text=f"Measurement: {measurement_display}",
            )
            figures.append(fig)

        return figures
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error regenerating experiment results figures: {e}")
        return []


def _regenerate_gp_predictions_figures(
    mlflow_run: str,
    iteration: str,
    clicked_points: dict[str, list[int]] | None = None,
    pinned_points: dict[str, bool] | None = None,
    measurement_renames: dict[str, str] | None = None,
    investigation_space_renames: dict[str, str] | None = None,
    objectives_bounds: dict[str, dict[str, float | None]] | None = None,
    pareto_mode: str = "Optimization",
    robustness_confidence_level: int | None = None,
    robustness_pareto_point: int | None = None,
    alternative_expected_errors: dict | None = None,
) -> list[go.Figure]:
    """Regenerate GP prediction figures with current state and renames."""
    try:
        investigation_space = InvestigationSpace.from_json(
            Path(f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/investigation_space.json")
        )

        meshgrid_csv = (
            f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/meshgrid_predictions.csv"
        )
        meshgrid_df = pd.read_csv(meshgrid_csv)

        try:
            experiments_df = pd.read_csv(
                f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/experiments.csv"
            )
        except FileNotFoundError:
            experiments_df = pd.DataFrame()

        try:
            # Load pareto front data based on selected mode
            if pareto_mode == "Manufacturing":
                pareto_df = load_manufacturing_df(mlflow_run, iteration)
                if pareto_df.empty:
                    # Error: Manufacturing mode selected but data not available
                    print(
                        "Warning: Manufacturing data not found for GP predictions export"
                    )
                    return []  # Return empty list if manufacturing data not found
            else:
                pareto_csv = (
                    f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}"
                    "/pareto_front_predictions.csv"
                )
                pareto_df = pd.read_csv(pareto_csv)
        except FileNotFoundError:
            pareto_df = pd.DataFrame()

        variables = investigation_space.variables
        var_names = [v.name for v in variables]

        var_pairs = []
        pair_variables = []
        for i in range(0, len(variables), 2):
            if i + 1 < len(variables):
                var_pairs.append((variables[i].name, variables[i + 1].name))
                pair_variables.append([variables[i], variables[i + 1]])

        if len(variables) < 2:
            return []

        numeric_cols = meshgrid_df.select_dtypes(include=[np.number]).columns.tolist()
        prediction_cols = [col for col in numeric_cols if col not in var_names]

        if not prediction_cols or not var_pairs:
            return []

        figures = []
        num_cols = len(var_pairs)

        var_to_filter: dict[Any, Any] = {}
        if clicked_points is None:
            clicked_points = {}
        if pinned_points is None:
            pinned_points = {}

        prediction_col_ranges = {}
        for pred_col in prediction_cols:
            if pred_col in meshgrid_df.columns:
                pred_min = meshgrid_df[pred_col].min()
                pred_max = meshgrid_df[pred_col].max()
                prediction_col_ranges[pred_col] = (pred_min, pred_max)

        for prediction_col in prediction_cols:
            fig = make_subplots(
                rows=1,
                cols=num_cols,
                specs=[[{"secondary_y": False} for _ in range(num_cols)]],
                horizontal_spacing=0.12,
            )
            fig.update_layout(template=get_theme())

            for col_idx, (var_x, var_y) in enumerate(var_pairs):
                col = col_idx + 1
                if var_x not in meshgrid_df.columns or var_y not in meshgrid_df.columns:
                    continue
                if prediction_col not in meshgrid_df.columns:
                    continue

                show_colorbar = col_idx == 0

                # Define objective columns once per pair
                objective_cols = []
                if not experiments_df.empty:
                    objective_cols = [
                        c for c in experiments_df.columns if c.startswith("obj:")
                    ]
                elif not pareto_df.empty:
                    objective_cols = [
                        c for c in pareto_df.columns if c.startswith("obj:")
                    ]

                try:
                    x_vals = np.asarray(meshgrid_df[var_x], dtype=np.float64)
                    y_vals = np.asarray(meshgrid_df[var_y], dtype=np.float64)
                    z_vals = np.asarray(meshgrid_df[prediction_col], dtype=np.float64)

                    unique_x = np.unique(x_vals)
                    unique_y = np.unique(y_vals)
                    try:
                        z_grid = z_vals.reshape(len(unique_x), len(unique_y)).T
                    except ValueError:
                        pivot_df = meshgrid_df.pivot_table(
                            index=var_y, columns=var_x, values=prediction_col
                        )
                        z_grid = pivot_df.values
                        unique_x = pivot_df.columns.values
                        unique_y = pivot_df.index.values

                    z_min, z_max = prediction_col_ranges.get(
                        prediction_col, (np.nanmin(z_grid), np.nanmax(z_grid))
                    )

                    x_label = get_display_name(var_x, investigation_space_renames)
                    y_label = get_display_name(var_y, investigation_space_renames)
                    z_label = get_display_name(prediction_col, measurement_renames)

                    fig.add_trace(
                        go.Contour(
                            z=z_grid,
                            x=unique_x,
                            y=unique_y,
                            # colorscale="Viridis",
                            showscale=show_colorbar,
                            zmin=z_min,
                            zmax=z_max,
                            colorbar=dict(title=""),
                            contours=dict(showlabels=True, labelfont=dict(size=8)),
                            hovertemplate=(
                                f"<b>{x_label}</b>: %{{x:.4f}}<br>"
                                f"<b>{y_label}</b>: %{{y:.4f}}<br>"
                                f"<b>{z_label}</b>: %{{z:.4f}}<extra></extra>"
                            ),
                            name=f"Mean {z_label}",
                            showlegend=True,
                        ),
                        row=1,
                        col=col,
                    )

                    if (
                        not experiments_df.empty
                        and var_x in experiments_df.columns
                        and var_y in experiments_df.columns
                    ):
                        inside_bounds: list[Any] = []
                        outside_bounds: list[Any] = []

                        # Classify experiments as inside or outside bounds
                        for idx in experiments_df.index:
                            is_outside = False
                            # Only check bounds if we have objective columns and bounds dict
                            if isinstance(objectives_bounds, dict) and objective_cols:
                                for obj_col in objective_cols:
                                    # Check if this objective has bounds set
                                    if obj_col in objectives_bounds:
                                        bounds = objectives_bounds[obj_col]
                                        value = experiments_df.loc[idx, obj_col]
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

                                        # Check if outside any bound
                                        if (
                                            min_bound is not None
                                            and value_float < min_bound
                                        ):
                                            is_outside = True
                                            break
                                        if (
                                            max_bound is not None
                                            and value_float > max_bound
                                        ):
                                            is_outside = True
                                            break

                            (outside_bounds if is_outside else inside_bounds).append(
                                idx
                            )

                        if inside_bounds:
                            inside_exp_df = experiments_df.iloc[inside_bounds]
                            fig.add_trace(
                                go.Scatter(
                                    x=inside_exp_df[var_x],
                                    y=inside_exp_df[var_y],
                                    mode="markers",
                                    marker=dict(
                                        size=8,
                                        symbol=MARKER_SYMBOLS["experiment"],
                                        color=MARKER_COLORS["experiment"],
                                        line=dict(width=1.5, color="white"),
                                    ),
                                    name=MARKER_NAMES["experiment"],
                                    hovertemplate="<b>Experiment</b><extra></extra>",
                                    showlegend=True,
                                ),
                                row=1,
                                col=col,
                            )

                        if outside_bounds:
                            outside_exp_df = experiments_df.iloc[outside_bounds]
                            fig.add_trace(
                                go.Scatter(
                                    x=outside_exp_df[var_x],
                                    y=outside_exp_df[var_y],
                                    mode="markers",
                                    marker=dict(
                                        size=8,
                                        symbol=MARKER_SYMBOLS["experiment"],
                                        color=MARKER_COLORS["discarded"],
                                        line=dict(width=1.5, color="white"),
                                    ),
                                    name=MARKER_NAMES["discarded"],
                                    hovertemplate="<b>Experiment (Excluded)</b><extra></extra>",
                                    showlegend=True,
                                ),
                                row=1,
                                col=col,
                            )

                    x_min = pair_variables[col_idx][0].min
                    x_max = pair_variables[col_idx][0].max
                    y_min = pair_variables[col_idx][1].min
                    y_max = pair_variables[col_idx][1].max

                    fig.update_xaxes(
                        title_text=x_label,
                        row=1,
                        col=col,
                        range=[x_min, x_max],
                    )
                    fig.update_yaxes(
                        title_text=y_label,
                        row=1,
                        col=col,
                        range=[y_min, y_max],
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(f"Error processing pair {var_x} vs {var_y}: {e}")
                    continue

                # Add pareto points with bounds filtering (outside try-except)
                if (
                    not pareto_df.empty
                    and var_x in pareto_df.columns
                    and var_y in pareto_df.columns
                ):
                    # Classify pareto points as inside or outside bounds
                    pareto_inside_bounds: list[Any] = []
                    pareto_outside_bounds: list[Any] = []

                    for idx in pareto_df.index:
                        is_outside = False
                        # Only check bounds if we have objective columns and bounds dict
                        if isinstance(objectives_bounds, dict) and objective_cols:
                            for obj_col in objective_cols:
                                # Check if this objective has bounds set
                                if obj_col in objectives_bounds:
                                    bounds = objectives_bounds[obj_col]
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

                                    # Check if outside any bound
                                    if (
                                        min_bound is not None
                                        and value_float < min_bound
                                    ):
                                        is_outside = True
                                        break
                                    if (
                                        max_bound is not None
                                        and value_float > max_bound
                                    ):
                                        is_outside = True
                                        break

                        (
                            pareto_outside_bounds
                            if is_outside
                            else pareto_inside_bounds
                        ).append(idx)

                    # Add inside bounds pareto points
                    if pareto_inside_bounds:
                        inside_pareto_df = pareto_df.loc[pareto_inside_bounds]
                        fig.add_trace(
                            go.Scatter(
                                x=inside_pareto_df[var_x],
                                y=inside_pareto_df[var_y],
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    symbol=MARKER_SYMBOLS["pareto"],
                                    color=MARKER_COLORS["pareto"],
                                    line=dict(width=2, color="black"),
                                ),
                                hovertemplate="<b>Pareto Point</b><extra></extra>",
                                name=MARKER_NAMES["pareto"],
                                showlegend=True,
                            ),
                            row=1,
                            col=col,
                        )

                    # Add outside bounds pareto points (discarded)
                    if pareto_outside_bounds:
                        outside_pareto_df = pareto_df.loc[pareto_outside_bounds]
                        fig.add_trace(
                            go.Scatter(
                                x=outside_pareto_df[var_x],
                                y=outside_pareto_df[var_y],
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    symbol=MARKER_SYMBOLS["pareto"],
                                    color=MARKER_COLORS["discarded"],
                                    line=dict(width=2, color="black"),
                                ),
                                hovertemplate="<b>Pareto Point (Excluded)</b><extra></extra>",
                                name=MARKER_NAMES["discarded"],
                                showlegend=True,
                            ),
                            row=1,
                            col=col,
                        )

                    clicked_indices_gp = (
                        clicked_points.get("pareto-front", [])
                        if isinstance(clicked_points, dict)
                        else []
                    )
                    if clicked_indices_gp:
                        selected_x_gp = []
                        selected_y_gp = []
                        for idx in clicked_indices_gp:
                            if isinstance(idx, int) and idx < len(pareto_df):
                                selected_x_gp.append(pareto_df[var_x].iloc[idx])
                                selected_y_gp.append(pareto_df[var_y].iloc[idx])

                        if selected_x_gp:
                            fig.add_trace(
                                go.Scatter(
                                    x=selected_x_gp,
                                    y=selected_y_gp,
                                    mode="markers",
                                    marker=dict(
                                        size=12,
                                        symbol=MARKER_SYMBOLS["selected"],
                                        color=MARKER_COLORS["selected"],
                                        line=dict(width=2, color="black"),
                                    ),
                                    hovertemplate="<b>Selected Pareto Point</b><extra></extra>",
                                    name=MARKER_NAMES["selected"],
                                    showlegend=True,
                                ),
                                row=1,
                                col=col,
                            )

                    if pinned_points and isinstance(pinned_points, dict):
                        pinned_x_gp = []
                        pinned_y_gp = []
                        for idx_str in pinned_points.keys():
                            if pinned_points[idx_str]:
                                try:
                                    idx = int(idx_str)
                                    if idx < len(pareto_df):
                                        pinned_x_gp.append(pareto_df[var_x].iloc[idx])
                                        pinned_y_gp.append(pareto_df[var_y].iloc[idx])
                                except (ValueError, KeyError):
                                    pass

                        if pinned_x_gp:
                            fig.add_trace(
                                go.Scatter(
                                    x=pinned_x_gp,
                                    y=pinned_y_gp,
                                    mode="markers",
                                    marker=dict(
                                        size=12,
                                        symbol=MARKER_SYMBOLS["auto_selected"],
                                        color=MARKER_COLORS["auto_selected"],
                                        line=dict(width=2, color="black"),
                                    ),
                                    hovertemplate="<b>Auto Selection</b><extra></extra>",
                                    name=MARKER_NAMES["auto_selected"],
                                    showlegend=True,
                                ),
                                row=1,
                                col=col,
                            )

                    if (
                        robustness_pareto_point is not None
                        and robustness_confidence_level is not None
                        and not pareto_df.empty
                        and robustness_pareto_point < len(pareto_df)
                        and var_x in pareto_df.columns
                        and var_y in pareto_df.columns
                    ):
                        try:
                            pareto_point = pareto_df.iloc[robustness_pareto_point]
                            point_x = pareto_point[var_x]
                            point_y = pareto_point[var_y]

                            confidence_factors = {
                                25: 0.319,
                                50: 0.674,
                                75: 1.150,
                                80: 1.282,
                                85: 1.440,
                                90: 1.645,
                                95: 1.960,
                                99: 2.576,
                            }
                            z_score = confidence_factors.get(
                                robustness_confidence_level, 1.960
                            )

                            x_error = None
                            y_error = None
                            for var in investigation_space.variables:
                                if var.name == var_x and var.expected_error is not None:
                                    x_error = var.expected_error
                                elif (
                                    var.name == var_y and var.expected_error is not None
                                ):
                                    y_error = var.expected_error

                            if x_error is not None and y_error is not None:
                                x_interval = z_score * x_error
                                y_interval = z_score * y_error

                                rect_x = [
                                    point_x - x_interval,
                                    point_x + x_interval,
                                    point_x + x_interval,
                                    point_x - x_interval,
                                    point_x - x_interval,
                                ]
                                rect_y = [
                                    point_y - y_interval,
                                    point_y - y_interval,
                                    point_y + y_interval,
                                    point_y + y_interval,
                                    point_y - y_interval,
                                ]

                                fig.add_trace(
                                    go.Scatter(
                                        x=rect_x,
                                        y=rect_y,
                                        mode="lines",
                                        line=dict(color="red", width=2, dash="dash"),
                                        hovertemplate="<b>Confidence Box</b><br>"
                                        f"Confidence Level: "
                                        f"{robustness_confidence_level}%<extra></extra>",
                                        name=f"Confidence Box ({robustness_confidence_level}%)",
                                        showlegend=True,
                                        fill="toself",
                                        fillcolor="rgba(255,0,0,0.05)",
                                    ),
                                    row=1,
                                    col=col,
                                )

                                if alternative_expected_errors and (
                                    var_x in alternative_expected_errors
                                    or var_y in alternative_expected_errors
                                ):
                                    alt_x_error = alternative_expected_errors.get(
                                        var_x, x_error
                                    )
                                    alt_y_error = alternative_expected_errors.get(
                                        var_y, y_error
                                    )

                                    if (
                                        alt_x_error is not None
                                        and alt_y_error is not None
                                    ):
                                        alt_x_interval = z_score * alt_x_error
                                        alt_y_interval = z_score * alt_y_error
                                        alt_rect_x = [
                                            point_x - alt_x_interval,
                                            point_x + alt_x_interval,
                                            point_x + alt_x_interval,
                                            point_x - alt_x_interval,
                                            point_x - alt_x_interval,
                                        ]
                                        alt_rect_y = [
                                            point_y - alt_y_interval,
                                            point_y - alt_y_interval,
                                            point_y + alt_y_interval,
                                            point_y + alt_y_interval,
                                            point_y - alt_y_interval,
                                        ]

                                        fig.add_trace(
                                            go.Scatter(
                                                x=alt_rect_x,
                                                y=alt_rect_y,
                                                mode="lines",
                                                line=dict(
                                                    color="purple",
                                                    width=2,
                                                    dash="dash",
                                                ),
                                                hovertemplate=(
                                                    "<b>Alternative Confidence Box</b><br>"
                                                    "Confidence Level: "
                                                    f"{robustness_confidence_level}%<extra></extra>"
                                                ),
                                                name=(
                                                    f"Alternative Box "
                                                    f"({robustness_confidence_level}%)"
                                                ),
                                                showlegend=True,
                                                fill="toself",
                                                fillcolor="rgba(0,255,0,0.05)",
                                            ),
                                            row=1,
                                            col=col,
                                        )
                        except (KeyError, AttributeError, ValueError):
                            pass

                    if var_x in var_to_filter and var_y in var_to_filter:
                        selected_x_ref = var_to_filter[var_x]
                        selected_y_ref = var_to_filter[var_y]

                        fig.add_trace(
                            go.Scatter(
                                x=[selected_x_ref],
                                y=[selected_y_ref],
                                mode="markers",
                                marker=dict(
                                    size=12,
                                    color="red",
                                    symbol="diamond",
                                    line=dict(color="black", width=2),
                                ),
                                hovertemplate="<b>Reference</b><extra></extra>",
                                name="Reference",
                                showlegend=True,
                            ),
                            row=1,
                            col=col,
                        )

            pred_label = get_display_name(prediction_col, measurement_renames)
            fig.update_layout(
                height=500,
                showlegend=True,
                hovermode="closest",
                title_text=f"GP Prediction: {pred_label}",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.35,
                    entrywidth=300,
                ),
            )
            figures.append(fig)

        return figures
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error regenerating GP prediction figures: {e}")
        return []


def _export_investigation_space_csv(
    mlflow_run: str, investigation_space_renames: dict[str, str] | None = None
) -> str | None:
    """Export investigation space variables as CSV string."""
    try:
        path = Path(f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/investigation_space.json")
        investigation_space = InvestigationSpace.from_json(path)
        renames = investigation_space_renames or {}

        rows = []
        for var in investigation_space.variables:
            display_name = renames.get(var.name, var.name)
            rows.append(
                {
                    "name": display_name,
                    "min": var.min,
                    "max": var.max,
                    "current": getattr(var, "current", ""),
                }
            )

        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error exporting investigation space: {e}")
        return None


def _export_optimization_problem_csv(
    mlflow_run: str,
    objective_renames: dict[str, str] | None = None,
    pareto_mode: str = "Optimization",
) -> str | None:
    """Export optimization problem objectives as CSV string."""
    try:
        opt_problem = load_optimization_problem(mlflow_run, pareto_mode)
        renames = objective_renames or {}

        rows = []
        for idx, axis in enumerate(opt_problem.problem):
            display_name = renames.get(str(idx), axis.var_name)
            rows.append(
                {
                    "objective_index": idx,
                    "variable_name": display_name,
                    "direction": "minimize" if axis.minimize else "maximize",
                }
            )

        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error exporting optimization problem: {e}")
        return None


def _export_measurements_csv(
    mlflow_run: str,
    iteration: str,
    investigation_space_renames: dict[str, str] | None = None,
    measurement_renames: dict[str, str] | None = None,
) -> str | None:
    """Export measurements (experiments) as CSV string with renamed columns."""
    try:
        path = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/experiments.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            inv_renames = investigation_space_renames or {}
            meas_renames = measurement_renames or {}

            rename_map = {}
            for col in df.columns:
                if col in inv_renames:
                    rename_map[col] = inv_renames[col]
                elif col in meas_renames:
                    rename_map[col] = meas_renames[col]

            if rename_map:
                df = df.rename(columns=rename_map)

            return df.to_csv(index=False)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error exporting measurements: {e}")
    return None


def _export_pareto_front_csv(
    mlflow_run: str,
    iteration: str,
    measurement_renames: dict[str, str] | None = None,
) -> str | None:
    """Export pareto front predictions as CSV string with renamed columns."""
    try:
        path = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/pareto_front_predictions.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            meas_renames = measurement_renames or {}

            rename_map = {}
            for col in df.columns:
                if col in meas_renames:
                    rename_map[col] = meas_renames[col]

            if rename_map:
                df = df.rename(columns=rename_map)

            return df.to_csv(index=False)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error exporting pareto front: {e}")
    return None


@app.callback(
    Output("download-plots-tables", "data"),
    Input("download-button", "n_clicks"),
    State("mlflow-run", "value"),
    State("iteration", "value"),
    State("clicked-points-store", "data"),
    State("objectives-bounds-store", "data"),
    State("pinned-points-store", "data"),
    State("objective-renames-store", "data"),
    State("investigation-space-renames-store", "data"),
    State("measurement-renames-store", "data"),
    State("gp-predictions-figures-store", "data"),
    State("robustness-confidence-level", "value"),
    State("robustness-pareto-point", "value"),
    State("alternative-expected-error-store", "data"),
    State("parcoords-columns-checklist", "value"),
    State("parcoords-visibility-checklist", "value"),
    State("pareto-mode-store", "data"),
    prevent_initial_call=True,
)
def download_current_plots(
    n_clicks: int | None,
    mlflow_run: str | None,
    iteration: str | None,
    clicked_points: dict[str, list[int]],
    objectives_bounds: dict[str, dict[str, float | None]],
    pinned_points: dict[str, bool] | None,
    objective_renames: dict[str, str],
    investigation_space_renames: dict[str, str],
    measurement_renames: dict[str, str],
    gp_prediction_figures_store: list[dict] | None,
    robustness_confidence_level: int | None,
    robustness_pareto_point: int | None,
    alternative_expected_errors: dict | None,
    selected_parcoords_columns: list[str] | None,
    parcoords_visibility: list[str] | None,
    pareto_mode: str,
) -> dict[str, str] | Any:
    """Download the current figures as a zip archive of PNGs with current state and renames."""
    if n_clicks is None:
        return no_update

    figures: list[tuple[str, go.Figure]] = []

    if mlflow_run is not None and iteration is not None:
        exp_figures = _regenerate_experiment_results_figures(
            mlflow_run,
            iteration,
            clicked_points,
            measurement_renames,
            investigation_space_renames,
        )
        for index, fig in enumerate(exp_figures, start=1):
            figures.append((f"experiment-results-{index:02d}", fig))

        investigation_space_figure = update_investigation_space_plots(
            mlflow_run,
            iteration,
            clicked_points,
            pinned_points,
            investigation_space_renames,
            measurement_renames,
            objective_renames,
            objectives_bounds,
            pareto_mode,
        )
        figures.append(("investigation-space", investigation_space_figure))

        # Always regenerate GP prediction figures with current selection state
        # so that selected/pinned points are included in the exported plots.
        gp_figures = _regenerate_gp_predictions_figures(
            mlflow_run,
            iteration,
            clicked_points,
            pinned_points,
            measurement_renames,
            investigation_space_renames,
            objectives_bounds,
            pareto_mode,
            robustness_confidence_level,
            robustness_pareto_point,
            alternative_expected_errors,
        )

        # Fall back to stored serialized figures only if regeneration produced no figures
        if (not gp_figures or len(gp_figures) == 0) and gp_prediction_figures_store:
            gp_figures = [
                _figure_from_store(figure_data)
                for figure_data in gp_prediction_figures_store
            ]
        for index, fig in enumerate(gp_figures, start=1):
            figures.append((f"gp-predictions-{index:02d}", fig))

        pareto_figure = update_pareto_front_plots(
            mlflow_run,
            iteration,
            clicked_points,
            objectives_bounds,
            pinned_points or {},
            objective_renames or {},
            investigation_space_renames or {},
            measurement_renames or {},
            pareto_mode,
        )
        figures.append(("pareto-front", pareto_figure))

        # Add experiment results parallel coordinates plot
        exp_parcoords_figure = update_experiment_results_parcoords(
            mlflow_run,
            iteration,
            clicked_points,
            measurement_renames or {},
            investigation_space_renames or {},
        )
        figures.append(("experiment-results-parcoords", exp_parcoords_figure))

        # Add Pareto front parallel coordinates plot
        pareto_parcoords_figure = update_pareto_front_parcoords(
            mlflow_run,
            iteration,
            clicked_points,
            pinned_points or {},
            objective_renames or {},
            investigation_space_renames or {},
            measurement_renames or {},
            selected_parcoords_columns or [],
            parcoords_visibility or [],
            pareto_mode,
        )
        figures.append(("pareto-front-parcoords", pareto_parcoords_figure))

    if not figures:
        return no_update

    def _write_archive(buffer: BytesIO) -> None:
        use_html_fallback = False
        with zipfile.ZipFile(
            buffer, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for prefix, figure in figures:
                # Determine plot type from prefix
                if "experiment-results" in prefix and "parcoords" not in prefix:
                    plot_type = "experiment-results"
                elif "gp-predictions" in prefix:
                    plot_type = "gp-predictions"
                elif "pareto-front" in prefix and "parcoords" not in prefix:
                    plot_type = "pareto-front"
                elif "investigation-space" in prefix:
                    plot_type = "investigation-space"
                else:
                    plot_type = "default"

                width, height = _export_dimensions(figure, plot_type)
                # Update figure layout with export dimensions
                figure.update_layout(width=width, height=height)

                if not use_html_fallback:
                    try:
                        image_bytes = pio.to_image(
                            figure,
                            format="png",
                            width=width,
                            height=height,
                            scale=2,
                        )
                        archive.writestr(f"{prefix}.png", image_bytes)
                    except RuntimeError as e:
                        if "Chrome" in str(e) or "Kaleido" in str(e):
                            use_html_fallback = True
                            figure.update_layout(width=width, height=height)
                            html_str = pio.to_html(figure, include_plotlyjs="cdn")
                            archive.writestr(f"{prefix}.html", html_str)
                        else:
                            raise
                else:
                    figure.update_layout(width=width, height=height)
                    html_str = pio.to_html(figure, include_plotlyjs="cdn")
                    archive.writestr(f"{prefix}.html", html_str)

            if mlflow_run is not None:
                investigation_space_csv = _export_investigation_space_csv(
                    mlflow_run, investigation_space_renames
                )
                if investigation_space_csv:
                    archive.writestr("investigation_space.csv", investigation_space_csv)

                opt_problem_csv = _export_optimization_problem_csv(
                    mlflow_run, objective_renames, pareto_mode
                )
                if opt_problem_csv:
                    archive.writestr("optimization_problem.csv", opt_problem_csv)

                if iteration is not None:
                    measurements_csv = _export_measurements_csv(
                        mlflow_run,
                        iteration,
                        investigation_space_renames,
                        measurement_renames,
                    )
                    if measurements_csv:
                        archive.writestr("measurements.csv", measurements_csv)

                    pareto_csv = _export_pareto_front_csv(
                        mlflow_run, iteration, measurement_renames
                    )
                    if pareto_csv:
                        archive.writestr("pareto_front.csv", pareto_csv)

            readme = (
                "This archive contains exports of the current dashboard plots and tables.\n\n"
                "PLOTS:\n"
                "  - experiment-results-*.png: Experiment results scatter plots\n"
                "  - investigation-space.png: Investigation space scatter plot\n"
                "  - experiment-results-parcoords.png: Experiment results parallel coordinates "
                "plot\n"
                "  - gp-predictions-*.png: GP prediction contour plots\n"
                "  - pareto-front.png: Pareto front scatter plots\n"
                "  - pareto-front-parcoords.png: Pareto front parallel coordinates plot\n\n"
                if not use_html_fallback
                else (
                    "This archive contains interactive HTML exports of the current dashboard "
                    "plots.\n"
                    "Open any .html file in your browser to view and interact with the plot.\n\n"
                    "PLOTS:\n"
                    "  - experiment-results-*.html: Experiment results scatter plots\n"
                    "  - investigation-space.html: Investigation space scatter plot\n"
                    "  - experiment-results-parcoords.html: Experiment results parallel "
                    "coordinates plot\n"
                    "  - gp-predictions-*.html: GP prediction contour plots\n"
                    "  - pareto-front.html: Pareto front scatter plots\n"
                    "  - pareto-front-parcoords.html: Pareto front parallel coordinates plot\n\n"
                )
            )
            readme += (
                "TABLES:\n"
                "  - investigation_space.csv: Variables with min/max bounds\n"
                "  - optimization_problem.csv: Objectives and directions\n"
                "  - measurements.csv: Raw experiment data\n"
                "  - pareto_front.csv: Pareto front predictions\n"
            )
            archive.writestr("README.txt", readme)

    return dcc.send_bytes(_write_archive, "dashboard-plots.zip")
