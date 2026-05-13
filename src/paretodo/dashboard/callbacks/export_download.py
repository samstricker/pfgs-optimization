"""Callbacks for downloading and exporting plots/data."""

from io import BytesIO, StringIO
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

from paretodo.data_models import InvestigationSpace, OptimizationProblem

from ..server import app
from ..utils.types_and_constants import LOCAL_TEMP_DIR
from ..utils.plotting_utils import get_display_name
from ..utils.data_utils import (
    load_investigation_space,
    load_experiments_df,
    get_measurement_columns,
    get_variable_pairs,
)
from ..theme.plotly_theme import get_theme
from .pareto_front import update_pareto_front_plots

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


def _export_dimensions(fig: go.Figure) -> tuple[int, int]:
    """Compute consistent export dimensions for a figure."""
    layout = fig.to_plotly_json().get("layout", {})
    title = _figure_title(fig)
    columns = _subplot_domains(fig, "xaxis")
    rows = _subplot_domains(fig, "yaxis")

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
                            colorscale="Viridis",
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
    except Exception as e:
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
) -> list[go.Figure]:
    """Regenerate GP prediction figures with current state and renames."""
    try:
        import numpy as np
        from plotly.subplots import make_subplots

        investigation_space = InvestigationSpace.from_json(
            Path(f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/investigation_space.json")
        )

        meshgrid_csv = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/meshgrid_predictions.csv"
        meshgrid_df = pd.read_csv(meshgrid_csv)

        try:
            experiments_df = pd.read_csv(
                f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/experiments.csv"
            )
        except FileNotFoundError:
            experiments_df = pd.DataFrame()

        try:
            pareto_csv = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/pareto_front_predictions.csv"
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

        var_to_filter = {}
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
                            colorscale="Viridis",
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
                            name="GP Prediction",
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
                        objective_cols = [
                            c for c in experiments_df.columns if c.startswith("obj:")
                        ]
                        inside_bounds = []
                        outside_bounds = []

                        for idx in experiments_df.index:
                            is_outside = False
                            if objectives_bounds and objective_cols:
                                for obj_col in objective_cols:
                                    if obj_col in objectives_bounds:
                                        bounds = objectives_bounds[obj_col]
                                        value = experiments_df.loc[idx, obj_col]
                                        try:
                                            min_bound = (
                                                float(bounds.get("min"))
                                                if bounds.get("min") is not None
                                                else None
                                            )
                                            max_bound = (
                                                float(bounds.get("max"))
                                                if bounds.get("max") is not None
                                                else None
                                            )
                                            if not isinstance(value, (int, float)):
                                                continue
                                            if (
                                                min_bound is not None
                                                and value < min_bound
                                            ):
                                                is_outside = True
                                                break
                                            if (
                                                max_bound is not None
                                                and value > max_bound
                                            ):
                                                is_outside = True
                                                break
                                        except (ValueError, TypeError):
                                            continue

                            (outside_bounds if is_outside else inside_bounds).append(
                                idx
                            )

                        if inside_bounds:
                            fig.add_trace(
                                go.Scatter(
                                    x=experiments_df.loc[inside_bounds, var_x],
                                    y=experiments_df.loc[inside_bounds, var_y],
                                    mode="markers",
                                    marker=dict(
                                        size=10,
                                        symbol="square",
                                        color="#E885B4",
                                        line=dict(width=1, color="black"),
                                    ),
                                    showlegend=False,
                                    hoverinfo="skip",
                                ),
                                row=1,
                                col=col,
                            )

                        if outside_bounds:
                            fig.add_trace(
                                go.Scatter(
                                    x=experiments_df.loc[outside_bounds, var_x],
                                    y=experiments_df.loc[outside_bounds, var_y],
                                    mode="markers",
                                    marker=dict(
                                        size=10,
                                        symbol="x",
                                        color="black",
                                        line=dict(width=1, color="black"),
                                    ),
                                    showlegend=False,
                                    hoverinfo="skip",
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
                except Exception as e:
                    print(f"Error processing pair {var_x} vs {var_y}: {e}")
                    continue

                if (
                    not pareto_df.empty
                    and var_x in pareto_df.columns
                    and var_y in pareto_df.columns
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=pareto_df[var_x],
                            y=pareto_df[var_y],
                            mode="markers",
                            marker=dict(
                                size=10,
                                symbol="circle",
                                color="#648FFF",
                                line=dict(width=2, color="black"),
                            ),
                            hovertemplate="<b>Pareto Point</b><extra></extra>",
                            name="Pareto Front",
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
                                        symbol="x",
                                        color="#FFB000",
                                        line=dict(width=2, color="black"),
                                    ),
                                    hovertemplate="<b>Selected Pareto Point</b><extra></extra>",
                                    name="Selected Pareto Point",
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
                                        symbol="diamond",
                                        color="#FFA500",
                                        line=dict(width=2, color="black"),
                                    ),
                                    hovertemplate="<b>Auto Selection</b><extra></extra>",
                                    name="Auto Selection",
                                    showlegend=True,
                                ),
                                row=1,
                                col=col,
                            )

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
                    orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.2
                ),
            )
            figures.append(fig)

        return figures
    except Exception as e:
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
            rows.append({
                "name": display_name,
                "min": var.min,
                "max": var.max,
                "current": getattr(var, "current", ""),
            })
        
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
    except Exception as e:
        print(f"Error exporting investigation space: {e}")
        return None


def _export_optimization_problem_csv(
    mlflow_run: str, objective_renames: dict[str, str] | None = None
) -> str | None:
    """Export optimization problem objectives as CSV string."""
    try:
        path = Path(f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/optimization_problem.json")
        opt_problem = OptimizationProblem.from_json(path)
        renames = objective_renames or {}
        
        rows = []
        for idx, axis in enumerate(opt_problem.problem):
            display_name = renames.get(str(idx), axis.var_name)
            rows.append({
                "objective_index": idx,
                "variable_name": display_name,
                "direction": axis.direction,
            })
        
        df = pd.DataFrame(rows)
        return df.to_csv(index=False)
    except Exception as e:
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
    except Exception as e:
        print(f"Error exporting measurements: {e}")
    return None


def _export_pareto_front_csv(
    mlflow_run: str,
    iteration: str,
    objective_renames: dict[str, str] | None = None,
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
    except Exception as e:
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
):
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

        gp_figures = _regenerate_gp_predictions_figures(
            mlflow_run,
            iteration,
            clicked_points,
            pinned_points,
            measurement_renames,
            investigation_space_renames,
            objectives_bounds,
        )
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
        )
        figures.append(("pareto-front", pareto_figure))

    if not figures:
        return no_update

    def _write_archive(buffer: BytesIO) -> None:
        use_html_fallback = False
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for prefix, figure in figures:
                width, height = _export_dimensions(figure)
                
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
                    mlflow_run, objective_renames
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
                        mlflow_run, iteration, objective_renames, measurement_renames
                    )
                    if pareto_csv:
                        archive.writestr("pareto_front.csv", pareto_csv)
            
            readme = (
                "This archive contains exports of the current dashboard plots and tables.\n\n"
                "PLOTS:\n"
                "  - experiment-results-*.png: Experiment results scatter plots\n"
                "  - gp-predictions-*.png: GP prediction contour plots\n"
                "  - pareto-front.png: Pareto front scatter plots\n\n"
                if not use_html_fallback
                else (
                    "This archive contains interactive HTML exports of the current dashboard plots.\n"
                    "Open any .html file in your browser to view and interact with the plot.\n\n"
                    "PLOTS:\n"
                    "  - experiment-results-*.html: Experiment results scatter plots\n"
                    "  - gp-predictions-*.html: GP prediction contour plots\n"
                    "  - pareto-front.html: Pareto front scatter plots\n\n"
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
