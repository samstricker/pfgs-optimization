"""Callbacks for experiment results visualization."""

from dash import Input, Output, html, dcc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from ..server import app

from ..utils.data_utils import (
    load_investigation_space,
    load_experiments_df,
    get_measurement_columns,
    get_variable_pairs,
)
from ..utils.plotting_utils import get_display_name
from ..theme.plotly_theme import get_theme
from ..utils.types_and_constants import MARKER_COLORS, MARKER_SYMBOLS
from ..utils.types_and_constants import ClickedPointsData, RenamesDict

__all__ = [
    "update_experiment_results",
    "store_experiment_results_figures",
    "update_experiment_results_parcoords",
]


@app.callback(
    Output("experiment-results-container", "children"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    Input("clicked-points-store", "data"),
    Input("measurement-renames-store", "data"),
    Input("investigation-space-renames-store", "data"),
)
def update_experiment_results(
    mlflow_run: str,
    iteration: str,
    clicked_points: ClickedPointsData,
    measurement_renames: RenamesDict,
    investigation_space_renames: RenamesDict,
):
    """Create separate scatter plots for each measurement."""
    if mlflow_run is None or iteration is None:
        return [html.Div("No data selected")]

    try:
        investigation_space = load_investigation_space(mlflow_run)
        experiments_df = load_experiments_df(mlflow_run, iteration)
        variables = investigation_space.variables

        if len(variables) < 2:
            return [html.Div("Not enough variables to plot")]

        var_names = [v.name for v in variables]
        measurement_cols = get_measurement_columns(experiments_df, var_names)
        var_pairs = get_variable_pairs(variables)

        if not measurement_cols or not var_pairs:
            return [html.Div("Could not create plots")]

        selected_indices = (
            clicked_points.get("experiment-results", [])
            if isinstance(clicked_points, dict)
            else []
        )

        num_cols = len(var_pairs)
        graph_components = []

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
                        customdata=[[idx] for idx in experiments_df.index],  # type: ignore
                        hovertemplate=(
                            f"<b>{get_display_name(var_x, investigation_space_renames)}"
                            "</b>: %{x:.4f}<br>"
                            f"<b>{get_display_name(var_y, investigation_space_renames)}"
                            "</b>: %{y:.4f}<br>"
                            f"<b>{get_display_name(measurement_col, measurement_renames)}"
                            "</b>: %{marker.color:.4f}<br>"
                            "<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=1,
                    col=col,
                )

                selected_x = []
                selected_y = []
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
                                symbol=MARKER_SYMBOLS["highlight"],
                                color=MARKER_COLORS["highlight"],
                                line=dict(width=2, color="black"),
                            ),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=col,
                    )

                fig.update_xaxes(
                    title_text=get_display_name(var_x, investigation_space_renames),
                    row=1,
                    col=col,
                )
                fig.update_yaxes(
                    title_text=get_display_name(var_y, investigation_space_renames),
                    row=1,
                    col=col,
                )

            fig.update_layout(
                height=500,
                showlegend=False,
                hovermode="closest",
                title_text=f"Measurement: {get_display_name(measurement_col, measurement_renames)}",
            )

            graph_components.append(
                html.Div(
                    [
                        dcc.Graph(
                            id={
                                "type": "experiment-results-graph",
                                "index": measurement_col,
                            },
                            figure=fig,
                            style={"width": "100%"},
                        ),
                    ]
                )
            )

        return graph_components

    except FileNotFoundError as e:
        return [html.Div(f"Data not found: {e}")]
    except Exception as e:  # pylint: disable=broad-exception-caught
        return [html.Div(f"Error loading experiment results: {e}")]


@app.callback(
    Output("experiment-results-figures-store", "data", allow_duplicate=True),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    Input("clicked-points-store", "data"),
    Input("measurement-renames-store", "data"),
    Input("investigation-space-renames-store", "data"),
    prevent_initial_call=True,
)
def store_experiment_results_figures(
    mlflow_run: str,
    iteration: str,
    clicked_points: ClickedPointsData,
    _measurement_renames: RenamesDict,
    _investigation_space_renames: RenamesDict,
) -> list[dict]:
    """Store experiment result figures for download."""
    if mlflow_run is None or iteration is None:
        return []

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
                                symbol=MARKER_SYMBOLS["highlight"],
                                color=MARKER_COLORS["highlight"],
                                line=dict(width=2, color="black"),
                            ),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=col,
                    )

                fig.update_xaxes(title_text=var_x, row=1, col=col)
                fig.update_yaxes(title_text=var_y, row=1, col=col)

            fig.update_layout(template=get_theme(), height=500, showlegend=False)
            figures.append(fig.to_dict())

        return figures
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error storing experiment figures: {e}")
        return []


@app.callback(
    Output("experiment-results-parcoords-graph", "figure"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
    Input("clicked-points-store", "data"),
    Input("measurement-renames-store", "data"),
    Input("investigation-space-renames-store", "data"),
    prevent_initial_call=True,
)
def update_experiment_results_parcoords(
    mlflow_run: str,
    iteration: str,
    clicked_points: ClickedPointsData,
    measurement_renames: RenamesDict,
    investigation_space_renames: RenamesDict,
) -> go.Figure:
    """Create a parallel coordinates plot for experiment results."""
    if mlflow_run is None or iteration is None:
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        return fig

    try:
        investigation_space = load_investigation_space(mlflow_run)
        experiments_df = load_experiments_df(mlflow_run, iteration)
        
        variables = investigation_space.variables
        if len(variables) < 1:
            fig = go.Figure()
            fig.update_layout(template=get_theme())
            fig.add_annotation(text="No variables to display")
            return fig

        var_names = [v.name for v in variables]

        # Select all numeric columns from the dataframe
        numeric_cols = experiments_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            fig = go.Figure()
            fig.update_layout(template=get_theme())
            fig.add_annotation(text="No numeric columns to display")
            return fig

        # Get selected/clicked indices
        selected_indices = (
            clicked_points.get("experiment-results", [])
            if isinstance(clicked_points, dict)
            else []
        )
        selected_indices = [int(x) for x in selected_indices if x is not None]

        # Create dimensions for parallel coordinates plot
        dimensions = []
        for col in numeric_cols:
            label = get_display_name(col, investigation_space_renames or measurement_renames)
            try:
                min_val = experiments_df[col].min()
                max_val = experiments_df[col].max()
                dimensions.append(
                    dict(
                        label=label,
                        values=experiments_df[col],
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

        # Create color array for selected vs non-selected points
        colors = []
        for idx in range(len(experiments_df)):
            if idx in selected_indices:
                colors.append(MARKER_COLORS["highlight"])
            else:
                colors.append(MARKER_COLORS["pareto"])

        # Create parallel coordinates plot
        fig = go.Figure(
            data=go.Parcoords(
                dimensions=dimensions,
                line=dict(
                    color=colors,
                    colorscale=None,
                    showscale=False,
                ),
                labelangle=-45,
                labelside="bottom",
            )
        )

        fig.update_layout(
            template=get_theme(),
            height=700,
            title="Experiment Results - Parallel Coordinates",
            font=dict(size=12),
            margin=dict(b=150),
        )

        return fig

    except FileNotFoundError as e:
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        fig.add_annotation(text=f"Data not found: {e}")
        return fig
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error creating experiment results parallel coordinates: {e}")
        fig = go.Figure()
        fig.update_layout(template=get_theme())
        fig.add_annotation(text=f"Error: {e}")
        return fig
