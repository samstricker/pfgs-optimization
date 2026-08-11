"""Consistent Plotly theme for all visualizations in the Pareto optimization dashboard."""

import plotly.io as pio
import plotly.graph_objects as go

SQUARE_PLOT_SIZE = 500
SQUARE_PLOT_LEFT_MARGIN = 80
SQUARE_PLOT_RIGHT_MARGIN = 80
SQUARE_PLOT_TOP_MARGIN = 80
SQUARE_PLOT_BOTTOM_MARGIN = 70
GP_PREDICTION_RIGHT_MARGIN = 80

# Define a consistent theme for all Plotly plots with professional styling
plotly_theme = go.layout.Template(
    layout=go.Layout(
        font=dict(
            family="'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
            size=14,
            color="rgb(40, 40, 40)",
        ),
        title=dict(
            font=dict(
                size=18,
                color="rgb(20, 20, 20)",
                family="'Segoe UI', sans-serif",
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgb(240, 240, 240)",
            gridwidth=0.8,
            showline=True,
            linecolor="rgb(180, 180, 180)",
            linewidth=1.5,
            title=dict(font=dict(size=14, color="rgb(40, 40, 40)")),
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgb(240, 240, 240)",
            gridwidth=0.8,
            showline=True,
            linecolor="rgb(180, 180, 180)",
            linewidth=1.5,
            title=dict(font=dict(size=14, color="rgb(40, 40, 40)")),
            zeroline=False,
        ),
        plot_bgcolor="rgb(250, 250, 250)",
        paper_bgcolor="rgb(255, 255, 255)",
        # margin=dict(l=80, r=420, t=80, b=70),
        legend=dict(
            font=dict(size=12, color="rgb(40, 40, 40)"),
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="rgb(180, 180, 180)",
            borderwidth=1,
            x=1.15,
            y=0.99,
            xanchor="left",
            yanchor="top",
            entrywidth=300,
        ),
        hovermode="closest",
    )
)


def register_theme():
    """Register the custom Plotly theme globally."""
    pio.templates["pareto_theme"] = plotly_theme
    pio.templates.default = "pareto_theme"


def get_theme():
    """Get the custom Plotly theme for use with individual figures."""
    return plotly_theme


def get_square_plot_dimensions(
    columns: int = 1,
    rows: int = 1,
    right_margin: int = SQUARE_PLOT_RIGHT_MARGIN,
    horizontal_spacing: float = 0.0,
    vertical_spacing: float = 0.0,
) -> tuple[int, int]:
    """Return canvas dimensions that keep each subplot area square."""
    columns = max(columns, 1)
    rows = max(rows, 1)

    horizontal_gap_factor = max(1.0 - horizontal_spacing * max(columns - 1, 0), 0.1)
    vertical_gap_factor = max(1.0 - vertical_spacing * max(rows - 1, 0), 0.1)

    inner_width = SQUARE_PLOT_SIZE * columns / horizontal_gap_factor
    inner_height = SQUARE_PLOT_SIZE * rows / vertical_gap_factor

    width = int(inner_width + SQUARE_PLOT_LEFT_MARGIN + right_margin)
    height = int(inner_height + SQUARE_PLOT_TOP_MARGIN + SQUARE_PLOT_BOTTOM_MARGIN)
    return width, height
