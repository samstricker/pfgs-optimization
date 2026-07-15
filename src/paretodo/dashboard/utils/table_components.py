"""Table generation components for displaying data in the Pareto visualization app."""

import base64
from typing import Optional, Literal
from dash import html, dcc
import pandas as pd

from paretodo.data_models import InvestigationSpace, OptimizationProblem
from .data_utils import load_pareto_df, load_manufacturing_df
from .types_and_constants import (
    TABLE_HEADER_STYLE,
    TABLE_CELL_STYLE,
    TABLE_ROW_ALT_BG,
    TABLE_ROW_NORMAL_BG,
    RenamesDict,
)

def _smiles_to_png_uri(smiles: str, size=(200, 160)) -> Optional[str]:
    """Render a SMILES string to a base64 PNG data URI, or None if invalid."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    encoded = base64.b64encode(drawer.GetDrawingText()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def create_table_header(headers: list[str], **kwargs: dict) -> html.Tr:
    """Create a table header row with consistent styling."""
    cells = []
    for header in headers:
        style = {**TABLE_HEADER_STYLE, **kwargs.get("style", {})}
        cells.append(html.Th(header, style=style))
    return html.Tr(cells)


def create_table_cell(
    content,
    idx: int,
    is_input: bool = False,
    input_id: Optional[dict] = None,
    input_type: Literal[
        "text", "number", "password", "email", "range", "search", "tel", "url", "hidden"
    ] = "text",
    min_width: str = "100px",
) -> html.Td:
    """Create a table cell with consistent styling."""
    row_style = {
        "backgroundColor": TABLE_ROW_ALT_BG if idx % 2 == 0 else TABLE_ROW_NORMAL_BG
    }
    cell_style = {**TABLE_CELL_STYLE, **row_style, "minWidth": min_width}

    if is_input and input_id:
        cell_content = dcc.Input(
            id=input_id,
            type=input_type,
            placeholder="Enter new name...",
            value=content or "",
            style={
                "width": "100%",
                "padding": "8px",
                "border": "1px solid #bdc3c7",
                "borderRadius": "4px",
                "fontSize": "14px",
                "boxSizing": "border-box",
            },
        )
        cell_style = {
            "border": "1px solid #bdc3c7",
            "padding": "8px",
            "minWidth": min_width,
            **row_style,
        }
    else:
        cell_content = str(content)  # type: ignore

    return html.Td(cell_content, style=cell_style)  # type: ignore


def create_investigation_space_table(
    investigation_space: InvestigationSpace,
    investigation_space_renames: Optional[RenamesDict] = None,
) -> html.Div:
    """Create table for investigation space with rename inputs."""
    if investigation_space_renames is None:
        investigation_space_renames = {}

    table_rows = [
        create_table_header(
            [
                "Variable Name",
                "Rename",
                "Min",
                "Max",
                "Discretization",
                "Expected Error",
                "Alternative Expected Error",
            ]
        )
    ]

    for idx, var in enumerate(investigation_space.variables):
        renamed_value = investigation_space_renames.get(var.name, "")
        discretization = (
            str(var.discretization) if var.discretization is not None else "N/A"
        )
        expected_error = (
            str(var.expected_error) if var.expected_error is not None else "N/A"
        )

        table_rows.append(
            html.Tr(
                [
                    create_table_cell(var.name, idx),
                    create_table_cell(
                        renamed_value,
                        idx,
                        is_input=True,
                        input_id={
                            "type": "investigation-space-rename-input",
                            "index": var.name,
                        },
                    ),
                    create_table_cell(str(var.min), idx),
                    create_table_cell(str(var.max), idx),
                    create_table_cell(discretization, idx),
                    create_table_cell(expected_error, idx),
                    create_table_cell(
                        "",
                        idx,
                        is_input=True,
                        input_id={
                            "type": "alternative-expected-error-input",
                            "index": var.name,
                        },
                    ),
                ]
            )
        )

    table = html.Table(
        table_rows,
        style={
            "borderCollapse": "collapse",
            "width": "100%",
            "marginTop": "20px",
            "fontFamily": "Arial, sans-serif",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
    )
    return html.Div([table])


def create_measurements_table(
    investigation_space_var_names: list[str],
    experiments_df: pd.DataFrame,
    measurement_renames: Optional[RenamesDict] = None,
) -> html.Div:
    """Create table for measurements with rename inputs."""
    if measurement_renames is None:
        measurement_renames = {}

    numeric_cols = experiments_df.select_dtypes(include="number").columns.tolist()
    measurement_cols = [
        col for col in numeric_cols if col not in investigation_space_var_names
    ]

    if not measurement_cols:
        return html.Div(
            "No measurements found.", style={"padding": "20px", "fontStyle": "italic"}
        )

    table_rows = [create_table_header(["Measurement Name", "Rename"])]

    for idx, col_name in enumerate(measurement_cols):
        renamed_value = measurement_renames.get(col_name, "")
        table_rows.append(
            html.Tr(
                [
                    create_table_cell(col_name, idx),
                    create_table_cell(
                        renamed_value,
                        idx,
                        is_input=True,
                        input_id={
                            "type": "measurement-rename-input",
                            "index": col_name,
                        },
                    ),
                ]
            )
        )

    table = html.Table(
        table_rows,
        style={
            "borderCollapse": "collapse",
            "width": "100%",
            "marginTop": "20px",
            "fontFamily": "Arial, sans-serif",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
    )
    return html.Div([table])


def create_optimization_problem_table(
    optimization_problem: OptimizationProblem,
    mlflow_run: Optional[str] = None,
    iteration: Optional[str] = None,
    pareto_mode: str = "Optimization",
) -> html.Div:
    """Create table for optimization problem with rename inputs."""
    pareto_col_names = []
    if mlflow_run and iteration:
        try:
            # Load the appropriate Pareto dataframe based on mode
            if pareto_mode == "Manufacturing":
                pareto_df = load_manufacturing_df(mlflow_run, iteration)
            else:
                pareto_df = load_pareto_df(mlflow_run, iteration)

            obj_columns = [col for col in pareto_df.columns if col.startswith("obj:")]
            pareto_col_names = [col.replace("obj:", "") for col in obj_columns]
        except FileNotFoundError:
            pass

    table_rows = [
        create_table_header(
            [
                "Objective Name",
                "Rename",
                "Minimize",
                "Optimization Type",
                "Lower Bound",
                "Upper Bound",
            ]
        )
    ]

    for idx, axis in enumerate(optimization_problem.problem):
        default_name = (
            pareto_col_names[idx] if idx < len(pareto_col_names) else axis.var_name
        )

        table_rows.append(
            html.Tr(
                [
                    create_table_cell(axis.var_name, idx),
                    create_table_cell(
                        default_name,
                        idx,
                        is_input=True,
                        input_id={"type": "objective-rename-input", "index": idx},
                        min_width="120px",
                    ),
                    create_table_cell("Yes" if axis.minimize else "No", idx),
                    create_table_cell(axis.optimization_type, idx),
                    create_table_cell(
                        (
                            str(axis.lower_bound)
                            if axis.lower_bound is not None
                            else "None"
                        ),
                        idx,
                    ),
                    create_table_cell(
                        (
                            str(axis.upper_bound)
                            if axis.upper_bound is not None
                            else "None"
                        ),
                        idx,
                    ),
                ]
            )
        )

    table = html.Table(
        table_rows,
        style={
            "borderCollapse": "collapse",
            "width": "100%",
            "maxWidth": "1000px",
            "marginTop": "20px",
            "fontFamily": "Arial, sans-serif",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "borderRadius": "4px",
            "overflow": "hidden",
        },
    )
    return html.Div([table])


def create_objectives_bounds_table(
    objective_columns: list[str], objective_renames: Optional[RenamesDict] = None
) -> html.Div:
    """Create table for editing objective bounds."""
    if objective_renames is None:
        objective_renames = {}

    table_rows = [create_table_header(["Objective", "Min Value", "Max Value"])]

    for idx, obj_col in enumerate(objective_columns):
        display_name = obj_col.replace("obj:", "")
        display_name = objective_renames.get(str(idx), display_name)

        table_rows.append(
            html.Tr(
                [
                    create_table_cell(display_name, idx),
                    create_table_cell(
                        None,
                        idx,
                        is_input=True,
                        input_type="number",
                        input_id={"type": "objective-min-input", "index": obj_col},
                    ),
                    create_table_cell(
                        None,
                        idx,
                        is_input=True,
                        input_type="number",
                        input_id={"type": "objective-max-input", "index": obj_col},
                    ),
                ]
            )
        )

    table = html.Table(
        table_rows,
        style={
            "borderCollapse": "collapse",
            "width": "100%",
            "maxWidth": "700px",
            "marginBottom": "20px",
            "fontFamily": "Arial, sans-serif",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            "borderRadius": "4px",
            "overflow": "hidden",
        },
    )
    return html.Div([table])


def create_pareto_details_table(
    pareto_df: pd.DataFrame,
    points_to_show: dict[str, tuple[int, str]],
    objective_names_map: dict[int, str],
    objective_renames: Optional[RenamesDict] = None,
    investigation_space_renames: Optional[RenamesDict] = None,
    measurement_renames: Optional[RenamesDict] = None,
) -> html.Div:
    """Create table showing detailed values for selected Pareto points.

    Parameters
    ----------
    pareto_df : pd.DataFrame
        The Pareto front dataframe
    points_to_show : dict[str, tuple[int, str]]
        Dictionary mapping column labels to (index, selection_type) tuples
        where selection_type is "Manual" or "Auto"
    objective_names_map : dict[int, str]
        Mapping from objective index to objective name
    objective_renames : Optional[RenamesDict]
        Optional renames for objectives
    investigation_space_renames : Optional[RenamesDict]
        Optional renames for investigation space variables
    measurement_renames : Optional[RenamesDict]
        Optional renames for measurements
    """
    if objective_renames is None:
        objective_renames = {}
    if investigation_space_renames is None:
        investigation_space_renames = {}
    if measurement_renames is None:
        measurement_renames = {}

    table_rows = []
    header_cells = [
        html.Th(
            "Parameter",
            style={
                "border": "1px solid #bdc3c7",
                "padding": "12px",
                "textAlign": "left",
                "backgroundColor": "#34495e",
                "color": "white",
                "fontWeight": "bold",
                "minWidth": "150px",
            },
        )
    ]

    for label in points_to_show.keys():
        header_cells.append(
            html.Th(
                label,
                style={
                    "border": "1px solid #bdc3c7",
                    "padding": "12px",
                    "textAlign": "center",
                    "backgroundColor": "#34495e",
                    "color": "white",
                    "fontWeight": "bold",
                    "minWidth": "150px",
                },
            )
        )

    table_rows.append(html.Tr(header_cells))

    # Add selection type row
    selection_type_cells = [
        html.Td(
            "Selection Type",
            style={
                "border": "1px solid #bdc3c7",
                "padding": "12px",
                "fontWeight": "500",
                "backgroundColor": "#ecf0f1",
                "fontStyle": "italic",
            },
        )
    ]

    for label, (idx, selection_type) in points_to_show.items():
        selection_type_cells.append(
            html.Td(
                selection_type,
                style={
                    "border": "1px solid #bdc3c7",
                    "padding": "12px",
                    "textAlign": "center",
                    "backgroundColor": "#ecf0f1",
                    "fontStyle": "italic",
                    "fontWeight": "500",
                },
            )
        )

    table_rows.append(html.Tr(selection_type_cells))

    for col_idx, col_name in enumerate(pareto_df.columns):
        row_style = {
            "backgroundColor": (
                TABLE_ROW_ALT_BG if col_idx % 2 == 0 else TABLE_ROW_NORMAL_BG
            )
        }

        display_name = col_name
        if col_name.startswith("obj:"):
            obj_col_index = [
                c for c in pareto_df.columns if c.startswith("obj:")
            ].index(col_name)
            display_name = objective_names_map.get(
                obj_col_index, col_name.replace("obj:", "")
            )
            if objective_renames:
                display_name = objective_renames.get(str(obj_col_index), display_name)
        else:
            # Apply measurement renames first if available
            if measurement_renames and col_name in measurement_renames:
                display_name = measurement_renames[col_name]
            # Otherwise apply investigation space renames for non-objective columns
            elif investigation_space_renames:
                display_name = investigation_space_renames.get(col_name, col_name)

        row_cells = [
            html.Td(
                display_name,
                style={
                    "border": "1px solid #bdc3c7",
                    "padding": "12px",
                    "fontWeight": "500",
                    **row_style,
                },
            )
        ]

        for label, (idx, selection_type) in points_to_show.items():
            if idx in pareto_df.index:
                value = pareto_df.loc[idx, col_name]
                formatted_value = (
                    f"{value:.6g}" if isinstance(value, (int, float)) else str(value)
                )
            else:
                formatted_value = "N/A"

            row_cells.append(
                html.Td(
                    formatted_value,
                    style={
                        "border": "1px solid #bdc3c7",
                        "padding": "12px",
                        "textAlign": "right",
                        "fontFamily": "monospace",
                        **row_style,
                    },
                )
            )

        table_rows.append(html.Tr(row_cells))

    # Structure row: render each selected point's SMILES with RDKit
    if "SMILES" in pareto_df.columns:
        structure_cells = [
            html.Td(
                "Structure",
                style={
                    "border": "1px solid #bdc3c7",
                    "padding": "12px",
                    "fontWeight": "500",
                    "backgroundColor": TABLE_ROW_ALT_BG,
                },
            )
        ]

        for label, (idx, selection_type) in points_to_show.items():
            cell_style = {
                "border": "1px solid #bdc3c7",
                "padding": "12px",
                "textAlign": "center",
                "backgroundColor": TABLE_ROW_ALT_BG,
            }

            if idx not in pareto_df.index:
                structure_cells.append(html.Td("N/A", style=cell_style))
                continue

            uri = _smiles_to_png_uri(str(pareto_df.loc[idx, "SMILES"]))
            if uri is None:
                structure_cells.append(html.Td("no render", style=cell_style))
            else:
                structure_cells.append(
                    html.Td(html.Img(src=uri, style={"maxWidth": "100%"}), style=cell_style)
                )

        table_rows.append(html.Tr(structure_cells))

    table = html.Table(
        table_rows,
        style={
            "borderCollapse": "collapse",
            "width": "100%",
            "marginTop": "20px",
            "fontFamily": "Arial, sans-serif",
            "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        },
    )
    return html.Div([table])
