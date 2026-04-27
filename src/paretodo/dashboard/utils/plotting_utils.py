"""Plotting utilities and common functions for Pareto visualization."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .types_and_constants import (
    MARKER_COLORS,
    MARKER_SYMBOLS,
    RenamesDict,
)


def create_marker_arrays(
    _df: pd.DataFrame,
    indices: list[int],
    selected_indices: list[int],
    outside_bounds_indices: list[int] | None = None,
) -> tuple[list[str], list[int], list[str]]:
    """Create marker color, size, and symbol arrays based on selection and bounds status."""
    if outside_bounds_indices is None:
        outside_bounds_indices = []

    colors = []
    sizes = []
    symbols = []

    for idx in indices:
        if idx in selected_indices:
            colors.append(MARKER_COLORS["selected"])
            symbols.append(MARKER_SYMBOLS["selected"])
            sizes.append(14)
        elif idx in outside_bounds_indices:
            colors.append(MARKER_COLORS["discarded"])
            symbols.append(MARKER_SYMBOLS["pareto"])
            sizes.append(10)
        else:
            colors.append(MARKER_COLORS["pareto"])
            symbols.append(MARKER_SYMBOLS["pareto"])
            sizes.append(10)

    return colors, sizes, symbols


def create_contour_data(
    df: pd.DataFrame, x_col: str, y_col: str, z_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create contour plot data from meshgrid dataframe."""
    x_vals = np.asarray(df[x_col], dtype=np.float64)
    y_vals = np.asarray(df[y_col], dtype=np.float64)
    z_vals = np.asarray(df[z_col], dtype=np.float64)

    unique_x = np.unique(x_vals)
    unique_y = np.unique(y_vals)

    try:
        z_grid = z_vals.reshape(len(unique_y), len(unique_x))
    except ValueError:
        pivot_df = df.pivot_table(index=y_col, columns=x_col, values=z_col)
        z_grid = pivot_df.values
        unique_x = pivot_df.columns.values
        unique_y = pivot_df.index.values

    return unique_x, unique_y, z_grid


def add_scatter_trace(
    fig: go.Figure,
    x_data: list,
    y_data: list,
    name: str = "",
    marker_color: str = MARKER_COLORS["pareto"],
    marker_symbol: str = MARKER_SYMBOLS["pareto"],
    marker_size: int = 8,
    show_legend: bool = False,
    row: int = 1,
    col: int = 1,
    hover_text: str = "",
) -> None:
    """Add scatter trace to figure."""
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            marker=dict(
                size=marker_size,
                symbol=marker_symbol,
                color=marker_color,
                line=dict(width=1.5, color="black"),
            ),
            name=name,
            hovertemplate=hover_text if hover_text else "",
            showlegend=show_legend,
        ),
        row=row,
        col=col,
    )


def add_contour_trace(
    fig: go.Figure,
    x_data: np.ndarray,
    y_data: np.ndarray,
    z_data: np.ndarray,
    colorscale: str = "Viridis",
    show_colorbar: bool = True,
    _color_title: str = "",
    row: int = 1,
    col: int = 1,
    z_min: float | None = None,
    z_max: float | None = None,
) -> None:
    """Add contour trace to figure."""
    if z_min is None:
        z_min = np.nanmin(z_data)
    if z_max is None:
        z_max = np.nanmax(z_data)

    fig.add_trace(
        go.Contour(
            z=z_data,
            x=x_data,
            y=y_data,
            colorscale=colorscale,
            showscale=show_colorbar,
            zmin=z_min,
            zmax=z_max,
            colorbar=dict(title=""),
            contours=dict(showlabels=True, labelfont=dict(size=8)),
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def get_display_name(var_name: str, renames: RenamesDict | None = None) -> str:
    """Get display name for a variable, applying renames if available."""
    if renames is None:
        return var_name
    return renames.get(var_name, var_name)


def build_hovertemplate(
    df: pd.DataFrame,
    objective_names_map: dict[int, str] | None = None,
    objective_renames: RenamesDict | None = None,
    investigation_space_renames: RenamesDict | None = None,
    measurement_renames: RenamesDict | None = None,
) -> str:
    """Build hovertemplate with proper display names and renames applied.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns to display in hover
    objective_names_map : dict[int, str] | None
        Mapping from objective index to objective name
    objective_renames : RenamesDict | None
        Optional renames for objectives
    investigation_space_renames : RenamesDict | None
        Optional renames for investigation space variables
    measurement_renames : RenamesDict | None
        Optional renames for measurements

    Returns
    -------
    str
        Hovertemplate string with renamed display names
    """
    hover_parts = []
    for j, col_name in enumerate(df.columns):
        display_name = col_name

        # Apply objective renames
        if col_name.startswith("obj:"):
            if objective_names_map:
                obj_col_index = [c for c in df.columns if c.startswith("obj:")].index(
                    col_name
                )
                obj_display = objective_names_map.get(
                    obj_col_index, col_name.replace("obj:", "")
                )
                if objective_renames and isinstance(objective_renames, dict):
                    obj_display = objective_renames.get(str(obj_col_index), obj_display)
                display_name = obj_display
        # Apply measurement renames
        elif (
            measurement_renames
            and isinstance(measurement_renames, dict)
            and col_name in measurement_renames
        ):
            display_name = measurement_renames[col_name]
        # Apply investigation space renames for remaining columns
        elif investigation_space_renames and isinstance(
            investigation_space_renames, dict
        ):
            display_name = investigation_space_renames.get(col_name, col_name)

        # Format value based on type
        if df[col_name].dtype in ["float64", "int64", "int32", "float32"]:
            hover_parts.append(f"<b>{display_name}</b>: %{{customdata[{j+1}]:.4f}}")
        else:
            hover_parts.append(f"<b>{display_name}</b>: %{{customdata[{j+1}]}}")

    return "<br>".join(hover_parts) + "<extra></extra>"


def create_hover_template(field_name: str, value_format: str = ".4f") -> str:
    """Create hover template string for a single field."""
    return f"<b>{field_name}</b>: %{{:{value_format}}}<br>"


def create_subplot_grid(num_plots: int, cols: int = 3) -> tuple[int, int]:
    """Calculate grid dimensions for subplots."""
    rows = (num_plots + cols - 1) // cols
    return rows, cols
