"""Consistent Plotly theme for all visualizations in the Pareto optimization dashboard."""

import plotly.io as pio
import plotly.graph_objects as go

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
        margin=dict(l=80, r=40, t=80, b=70),
        legend=dict(
            font=dict(size=12, color="rgb(40, 40, 40)"),
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="rgb(180, 180, 180)",
            borderwidth=1,
            x=1.02,
            y=0.99,
            xanchor="left",
            yanchor="top",
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
