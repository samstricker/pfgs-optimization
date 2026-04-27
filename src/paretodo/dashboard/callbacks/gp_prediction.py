"""Callbacks for GP predictions visualization."""

__all__ = [
    "populate_gp_filters",
    "update_gp_predictions",
]

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, dcc, html
from plotly.subplots import make_subplots

from paretodo.data_models import InvestigationSpace

from ..server import app
from ..utils.types_and_constants import (
    LOCAL_TEMP_DIR,
    ClickedPointsData,
    MARKER_SYMBOLS,
    MARKER_COLORS,
    MARKER_NAMES,
)
from ..theme.plotly_theme import get_theme


@app.callback(
    Output("gp-filters-container", "children"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    prevent_initial_call=True,
)
def populate_gp_filters(
    mlflow_run: str | None,
    iteration: str | None,
) -> list[html.Div]:
    """Create global filter dropdowns for variables not being plotted."""
    if mlflow_run is None or iteration is None:
        return []

    try:
        # Load investigation space
        investigation_space = InvestigationSpace.from_json(
            Path(f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/investigation_space.json")
        )

        # Load meshgrid predictions data
        meshgrid_csv = (
            f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/meshgrid_predictions.csv"
        )
        meshgrid_df = pd.read_csv(meshgrid_csv)

        variables = investigation_space.variables
        var_names = [v.name for v in variables]

        # Create dropdowns for all variables
        filter_controls = []
        for var_name in var_names:
            unique_vals = sorted(meshgrid_df[var_name].unique())
            filter_controls.append(
                html.Div(
                    [
                        html.Label(f"{var_name}:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id={"type": "gp-filter-dropdown", "index": var_name},
                            options=[  # type: ignore
                                {"label": f"{v:.4g}", "value": v} for v in unique_vals
                            ],
                            value=unique_vals[0],
                            style={"width": "150px"},
                        ),
                    ],
                    style={"display": "inline-block", "marginRight": "20px"},
                )
            )

        if filter_controls:
            return [
                html.Div(
                    filter_controls,
                    style={
                        "padding": "10px",
                        "backgroundColor": "#f5f5f5",
                        "borderRadius": "5px",
                        "marginBottom": "15px",
                    },
                )
            ]
        return []

    except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
        return [
            html.Div(
                f"Error loading filters: {str(e)}",
                style={"color": "red", "padding": "10px"},
            )
        ]


@app.callback(
    Output("gp-predictions-container", "children"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    Input({"type": "gp-filter-dropdown", "index": ALL}, "value"),
    Input("clicked-points-store", "data"),
    Input("pinned-points-store", "data"),
    Input("measurement-renames-store", "data"),
    Input("investigation-space-renames-store", "data"),
    Input("objectives-bounds-store", "data"),
    prevent_initial_call=True,
)
def update_gp_predictions(
    mlflow_run: str | None,
    iteration: str | None,
    filter_values: list[float],
    clicked_points: ClickedPointsData,
    pinned_points: dict[str, bool] | None,
    measurement_renames: dict[str, str],
    investigation_space_renames: dict[str, str],
    objectives_bounds: dict | None,
) -> list[html.Div]:
    """Create contour plots for each GP prediction using variable pair layout."""
    if mlflow_run is None or iteration is None:
        return [html.Div("No data selected")]

    try:
        # Load investigation space to get variable names
        investigation_space = InvestigationSpace.from_json(
            Path(f"{LOCAL_TEMP_DIR}/{mlflow_run}/general/investigation_space.json")
        )

        # Load meshgrid predictions data
        meshgrid_csv = (
            f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/meshgrid_predictions.csv"
        )
        meshgrid_df = pd.read_csv(meshgrid_csv)

        # Load experiments and Pareto front data for overlays
        experiments_csv = f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/experiments.csv"
        pareto_csv = (
            f"{LOCAL_TEMP_DIR}/{mlflow_run}/{iteration}/pareto_front_predictions.csv"
        )
        try:
            experiments_df = pd.read_csv(experiments_csv)
        except FileNotFoundError:
            experiments_df = pd.DataFrame()
        try:
            pareto_df = pd.read_csv(pareto_csv)
        except FileNotFoundError:
            pareto_df = pd.DataFrame()

        variables = investigation_space.variables
        var_names = [v.name for v in variables]

        # Create variable pairs (var1 vs var2, var3 vs var4, etc.)
        var_pairs = []
        pair_variables = []
        for i in range(0, len(variables), 2):
            if i + 1 < len(variables):
                var_pairs.append((variables[i].name, variables[i + 1].name))
                pair_variables.append([variables[i], variables[i + 1]])

        if len(variables) < 2:
            return [html.Div("Not enough variables to plot")]

        # Find prediction columns
        numeric_cols = meshgrid_df.select_dtypes(include=[np.number]).columns.tolist()
        prediction_cols = [col for col in numeric_cols if col not in var_names]

        if not prediction_cols:
            return [html.Div("No prediction columns found")]

        if not var_pairs:
            return [html.Div("Could not create variable pairs")]

        num_cols = len(var_pairs)

        # Create mapping of variable names to their selected filter values
        var_to_filter = {}
        if filter_values and len(filter_values) == len(var_names):
            var_to_filter = dict(zip(var_names, filter_values))

        # Create a separate figure for each prediction column
        graph_components = []

        # Calculate global min/max for each prediction column
        prediction_col_ranges = {}
        for pred_col in prediction_cols:
            if pred_col in meshgrid_df.columns:
                pred_min = meshgrid_df[pred_col].min()
                pred_max = meshgrid_df[pred_col].max()
                prediction_col_ranges[pred_col] = (pred_min, pred_max)

        for prediction_col in prediction_cols:
            # Create subplots for this prediction: 1 row, multiple columns
            fig = make_subplots(
                rows=1,
                cols=num_cols,
                specs=[[{"secondary_y": False} for _ in range(num_cols)]],
                horizontal_spacing=0.12,
            )
            fig.update_layout(template=get_theme())

            # Add contour traces for each variable pair
            for col_idx, (var_x, var_y) in enumerate(var_pairs):
                # Filter data for this specific pair using non-plotted variable values
                filtered_df = meshgrid_df.copy()

                # Find variables NOT in this pair and filter by their dropdown values
                plotted_in_pair = {var_x, var_y}
                for var_name in var_names:
                    if var_name not in plotted_in_pair and var_name in var_to_filter:
                        filter_val = var_to_filter[var_name]
                        filtered_df = filtered_df[filtered_df[var_name] == filter_val]

                col = col_idx + 1

                if var_x not in filtered_df.columns or var_y not in filtered_df.columns:
                    continue

                if prediction_col not in filtered_df.columns:
                    continue

                show_colorbar = col_idx == 0

                try:
                    # Create contour plot data
                    x_vals = np.asarray(filtered_df[var_x], dtype=np.float64)
                    y_vals = np.asarray(filtered_df[var_y], dtype=np.float64)
                    z_vals = np.asarray(filtered_df[prediction_col], dtype=np.float64)

                    # Get unique sorted values
                    unique_x = np.unique(x_vals)
                    unique_y = np.unique(y_vals)

                    # Reshape z values into a 2D grid
                    try:
                        z_grid = z_vals.reshape(len(unique_y), len(unique_x))
                    except ValueError:
                        # If reshape fails, try pivot table
                        pivot_df = filtered_df.pivot_table(
                            index=var_y, columns=var_x, values=prediction_col
                        )
                        z_grid = pivot_df.values
                        unique_x = pivot_df.columns.values
                        unique_y = pivot_df.index.values

                    # Get global min/max for consistent colorbar
                    z_min, z_max = prediction_col_ranges.get(
                        prediction_col, (np.nanmin(z_grid), np.nanmax(z_grid))
                    )

                    # Compute labels for hover template
                    x_label = (
                        investigation_space_renames.get(var_x, var_x)
                        if investigation_space_renames
                        else var_x
                    )
                    y_label = (
                        investigation_space_renames.get(var_y, var_y)
                        if investigation_space_renames
                        else var_y
                    )
                    z_label = (
                        measurement_renames.get(prediction_col, prediction_col)
                        if measurement_renames
                        else prediction_col
                    )

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
                            showlegend=False,
                        ),
                        row=1,
                        col=col,
                    )

                    # Add green squares for experiments (or black if outside bounds)
                    if (
                        not experiments_df.empty
                        and var_x in experiments_df.columns
                        and var_y in experiments_df.columns
                    ):
                        # Get objective columns from experiments_df
                        objective_cols = [
                            col
                            for col in experiments_df.columns
                            if col.startswith("obj:")
                        ]

                        # Separate experiments by outside bounds status
                        inside_bounds_indices = []
                        outside_bounds_indices = []

                        for idx in experiments_df.index:
                            is_outside_bounds = False
                            if (
                                objectives_bounds
                                and isinstance(objectives_bounds, dict)
                                and objective_cols
                            ):
                                for obj_col in objective_cols:
                                    if (
                                        obj_col in objectives_bounds
                                        and obj_col in experiments_df.columns
                                    ):
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
                                outside_bounds_indices.append(idx)
                            else:
                                inside_bounds_indices.append(idx)

                        # Add inside bounds experiments (green squares)
                        if inside_bounds_indices:
                            inside_exp_df = experiments_df.iloc[inside_bounds_indices]
                            fig.add_trace(
                                go.Scatter(
                                    x=inside_exp_df[var_x],
                                    y=inside_exp_df[var_y],
                                    mode="markers",
                                    marker=dict(
                                        size=8,
                                        symbol="square",
                                        color=MARKER_COLORS["experiment"],
                                        line=dict(width=1.5, color="white"),
                                    ),
                                    name=MARKER_NAMES["experiment"],
                                    hovertemplate="<b>Experiment</b><extra></extra>",
                                    showlegend=False,
                                ),
                                row=1,
                                col=col,
                            )

                        # Add outside bounds experiments (black squares)
                        if outside_bounds_indices:
                            outside_exp_df = experiments_df.iloc[outside_bounds_indices]
                            fig.add_trace(
                                go.Scatter(
                                    x=outside_exp_df[var_x],
                                    y=outside_exp_df[var_y],
                                    mode="markers",
                                    marker=dict(
                                        size=8,
                                        symbol="square",
                                        color=MARKER_COLORS["discarded"],
                                        line=dict(width=1.5, color="white"),
                                    ),
                                    name=MARKER_NAMES["discarded"],
                                    hovertemplate="<b>Experiment (Excluded)</b><extra></extra>",
                                    showlegend=False,
                                ),
                                row=1,
                                col=col,
                            )

                    # Add blue circles for Pareto front (or black if outside bounds)
                    if (
                        not pareto_df.empty
                        and var_x in pareto_df.columns
                        and var_y in pareto_df.columns
                    ):
                        # Get objective columns
                        objective_cols = [
                            col for col in pareto_df.columns if col.startswith("obj:")
                        ]

                        # Separate pareto points by outside bounds status
                        inside_bounds_indices = []
                        outside_bounds_indices = []

                        for idx in pareto_df.index:
                            is_outside_bounds = False
                            if objectives_bounds and isinstance(
                                objectives_bounds, dict
                            ):
                                for obj_col in objective_cols:
                                    if (
                                        obj_col in objectives_bounds
                                        and obj_col in pareto_df.columns
                                    ):
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
                                outside_bounds_indices.append(idx)
                            else:
                                inside_bounds_indices.append(idx)

                        # Add inside bounds Pareto points (blue circles)
                        if inside_bounds_indices:
                            inside_pareto_df = pareto_df.iloc[inside_bounds_indices]
                            fig.add_trace(
                                go.Scatter(
                                    x=inside_pareto_df[var_x],
                                    y=inside_pareto_df[var_y],
                                    mode="markers",
                                    marker=dict(
                                        size=8,
                                        symbol="circle",
                                        color=MARKER_COLORS["pareto"],
                                        line=dict(width=1.5, color="white"),
                                    ),
                                    name=MARKER_NAMES["pareto"],
                                    hovertemplate="<b>Pareto Point</b><extra></extra>",
                                    showlegend=False,
                                ),
                                row=1,
                                col=col,
                            )

                        # Add outside bounds Pareto points (black circles)
                        if outside_bounds_indices:
                            outside_pareto_df = pareto_df.iloc[outside_bounds_indices]
                            fig.add_trace(
                                go.Scatter(
                                    x=outside_pareto_df[var_x],
                                    y=outside_pareto_df[var_y],
                                    mode="markers",
                                    marker=dict(
                                        size=8,
                                        symbol="circle",
                                        color=MARKER_COLORS["discarded"],
                                        line=dict(width=1.5, color="white"),
                                    ),
                                    name=MARKER_NAMES["discarded"],
                                    hovertemplate="<b>Pareto Point (Excluded)</b><extra></extra>",
                                    showlegend=False,
                                ),
                                row=1,
                                col=col,
                            )

                    # Add orange crosses for selected experiments
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

                    if selected_exp_indices and not experiments_df.empty:
                        selected_exp_x = []
                        selected_exp_y = []
                        for idx in selected_exp_indices:
                            if (
                                idx < len(experiments_df)
                                and var_x in experiments_df.columns
                                and var_y in experiments_df.columns
                            ):
                                selected_exp_x.append(experiments_df[var_x].iloc[idx])
                                selected_exp_y.append(experiments_df[var_y].iloc[idx])

                        if selected_exp_x:
                            fig.add_trace(
                                go.Scatter(
                                    x=selected_exp_x,
                                    y=selected_exp_y,
                                    mode="markers",
                                    marker=dict(
                                        size=12,
                                        symbol=MARKER_SYMBOLS["highlight"],
                                        color=MARKER_COLORS["highlight"],
                                        line=dict(width=2, color="black"),
                                    ),
                                    hovertemplate="<b>Selected Experiment</b><extra></extra>",
                                    showlegend=False,
                                ),
                                row=1,
                                col=col,
                            )

                    # Add yellow X for selected Pareto points
                    selected_pareto_indices = []
                    if (
                        isinstance(clicked_points, dict)
                        and "pareto-front" in clicked_points
                    ):
                        selected_pareto_indices = clicked_points["pareto-front"]
                        if not isinstance(selected_pareto_indices, list):
                            selected_pareto_indices = (
                                list(selected_pareto_indices)
                                if selected_pareto_indices is not None
                                else []
                            )
                        selected_pareto_indices = [
                            x for x in selected_pareto_indices if x is not None
                        ]

                    if selected_pareto_indices and not pareto_df.empty:
                        selected_pareto_x = []
                        selected_pareto_y = []
                        for idx in selected_pareto_indices:
                            if (
                                idx < len(pareto_df)
                                and var_x in pareto_df.columns
                                and var_y in pareto_df.columns
                            ):
                                selected_pareto_x.append(pareto_df[var_x].iloc[idx])
                                selected_pareto_y.append(pareto_df[var_y].iloc[idx])

                        if selected_pareto_x:
                            fig.add_trace(
                                go.Scatter(
                                    x=selected_pareto_x,
                                    y=selected_pareto_y,
                                    mode="markers",
                                    marker=dict(
                                        size=12,
                                        symbol=MARKER_SYMBOLS["selected"],
                                        color=MARKER_COLORS["selected"],
                                        line=dict(width=2, color="black"),
                                    ),
                                    hovertemplate="<b>Selected Pareto Point</b><extra></extra>",
                                    showlegend=False,
                                ),
                                row=1,
                                col=col,
                            )

                    # Add black X for pinned Pareto points
                    if (
                        pinned_points
                        and isinstance(pinned_points, dict)
                        and not pareto_df.empty
                    ):
                        pinned_x = []
                        pinned_y = []
                        for idx_str in pinned_points.keys():
                            if pinned_points[idx_str]:
                                try:
                                    idx = int(idx_str)
                                    if (
                                        idx < len(pareto_df)
                                        and var_x in pareto_df.columns
                                        and var_y in pareto_df.columns
                                    ):
                                        pinned_x.append(pareto_df[var_x].iloc[idx])
                                        pinned_y.append(pareto_df[var_y].iloc[idx])
                                except (ValueError, KeyError):
                                    pass

                        if pinned_x:
                            fig.add_trace(
                                go.Scatter(
                                    x=pinned_x,
                                    y=pinned_y,
                                    mode="markers",
                                    marker=dict(
                                        size=12,
                                        symbol=MARKER_SYMBOLS["auto_selected"],
                                        color=MARKER_COLORS["auto_selected"],
                                        line=dict(width=2, color="black"),
                                    ),
                                    hovertemplate="<b>Auto Selection</b><extra></extra>",
                                    showlegend=False,
                                ),
                                row=1,
                                col=col,
                            )

                    # Add red diamond marker for selected point filter location
                    if var_x in var_to_filter and var_y in var_to_filter:
                        selected_x = var_to_filter[var_x]
                        selected_y = var_to_filter[var_y]

                        # Find the actual prediction value at selected point
                        selected_point_df = meshgrid_df.copy()
                        for var_name in var_names:
                            if var_name in var_to_filter:
                                val = var_to_filter[var_name]
                                selected_point_df = selected_point_df[
                                    selected_point_df[var_name] == val
                                ]

                        selected_z = None
                        if (
                            not selected_point_df.empty
                            and prediction_col in selected_point_df.columns
                        ):
                            selected_z = selected_point_df[prediction_col].iloc[0]

                        # Compute labels for hover text
                        x_hover = (
                            investigation_space_renames.get(var_x, var_x)
                            if investigation_space_renames
                            else var_x
                        )
                        y_hover = (
                            investigation_space_renames.get(var_y, var_y)
                            if investigation_space_renames
                            else var_y
                        )
                        hover_text = (
                            "<b>Selected Point</b><br>"
                            f"<b>{x_hover}</b>: {selected_x:.4f}<br>"
                            f"<b>{y_hover}</b>: {selected_y:.4f}<br>"
                        )
                        if selected_z is not None:
                            pred_label = (
                                prediction_col
                                if not measurement_renames
                                else measurement_renames.get(
                                    prediction_col, prediction_col
                                )
                            )
                            hover_text += f"<b>{pred_label}</b>: {selected_z:.4f}"

                        fig.add_trace(
                            go.Scatter(
                                x=[selected_x],
                                y=[selected_y],
                                mode="markers",
                                marker=dict(
                                    size=12,
                                    color="red",
                                    symbol="diamond",
                                    line=dict(color="black", width=2),
                                ),
                                hovertemplate=hover_text + "<extra></extra>",
                                showlegend=False,
                            ),
                            row=1,
                            col=col,
                        )

                except ValueError:
                    continue

                # Add axis labels
                display_var_x = (
                    investigation_space_renames.get(var_x, var_x)
                    if investigation_space_renames
                    else var_x
                )
                display_var_y = (
                    investigation_space_renames.get(var_y, var_y)
                    if investigation_space_renames
                    else var_y
                )

                # Get bounds from pair_variables
                x_min = pair_variables[col_idx][0].min
                x_max = pair_variables[col_idx][0].max
                y_min = pair_variables[col_idx][1].min
                y_max = pair_variables[col_idx][1].max

                fig.update_xaxes(
                    title_text=display_var_x,
                    row=1,
                    col=col,
                    range=[x_min, x_max],
                )
                fig.update_yaxes(
                    title_text=display_var_y,
                    row=1,
                    col=col,
                    range=[y_min, y_max],
                )

            pred_label = (
                prediction_col
                if not measurement_renames
                else measurement_renames.get(prediction_col, prediction_col)
            )
            fig.update_layout(
                height=500,
                showlegend=False,
                hovermode="closest",
                title_text=f"GP Prediction: {pred_label}",
            )

            # Create a graph component for this prediction
            graph_components.append(
                html.Div(
                    [
                        dcc.Graph(
                            id={
                                "type": "gp-prediction-graph",
                                "index": prediction_col,
                            },
                            figure=fig,
                            style={"width": "100%"},
                        ),
                    ]
                )
            )

        return graph_components if graph_components else [html.Div("No graphs created")]

    except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        if isinstance(e, FileNotFoundError):
            fig.add_annotation(text=f"Data not found: {e}")
        else:
            fig.add_annotation(text=f"Error loading GP predictions: {str(e)}")
        return [html.Div(dcc.Graph(figure=fig))]


# on Output decorator for gp-predictions-figures-store.data and other data stores
# to allow multiple callbacks to write to same store if needed
