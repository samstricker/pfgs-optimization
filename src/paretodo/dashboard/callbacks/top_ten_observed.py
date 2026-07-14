"""Callbacks: Top 10 observed molecules table with inline structures.

Reads the selected iteration's experiments.csv (columns RecipeID, SMILES,
LD50), sorts by LD50 descending, takes the top 10, renders each SMILES with
RDKit, and builds an HTML table with an inline clickable structure per row.

Clicking a structure opens a modal (an in-page overlay, same tab) showing that
molecule at 320x260, matching the pareto molecule-preview size.
"""

import base64
import json
import os

from dash import Input, Output, State, ALL, html, callback_context
from dash.exceptions import PreventUpdate

from ..server import app
from ..utils.types_and_constants import LOCAL_TEMP_DIR

__all__ = ["update_top_ten_table", "handle_structure_click"]

TOP_N = 10
THUMB_SIZE = (140, 110)
MODAL_SIZE = (320, 260)


def _smiles_to_png_uri(smiles: str, size) -> str | None:
    """Render SMILES to a base64 PNG data URI, or None if invalid/unavailable."""
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
    png = drawer.GetDrawingText()
    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.callback(
    Output("top-ten-table-container", "children"),
    Input("mlflow-run", "value"),
    Input("iteration", "value"),
)
def update_top_ten_table(mlflow_run, iteration):
    """Build the top-10 observed table for the selected iteration."""
    if mlflow_run is None or iteration is None:
        return html.Div("Select a run and iteration.")

    import pandas as pd

    csv_path = os.path.join(
        LOCAL_TEMP_DIR, mlflow_run, iteration, "experiments.csv"
    )
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return html.Div("experiments.csv not found for this iteration.")

    if "LD50" not in df.columns:
        return html.Div("experiments.csv has no LD50 column.")

    top = df.sort_values("LD50", ascending=False).head(TOP_N)

    header = html.Tr(
        [
            html.Th("RecipeID"),
            html.Th("SMILES"),
            html.Th("LD50"),
            html.Th("Structure"),
        ]
    )

    rows = []
    for _, row in top.iterrows():
        smiles = str(row["SMILES"])
        thumb = _smiles_to_png_uri(smiles, THUMB_SIZE)

        if thumb is None:
            cell = html.Td("no render")
        else:
            # Clicking the image fires handle_structure_click via pattern id.
            cell = html.Td(
                html.Img(
                    src=thumb,
                    id={"type": "top-ten-structure", "smiles": smiles},
                    n_clicks=0,
                    style={"cursor": "pointer"},
                )
            )

        rows.append(
            html.Tr(
                [
                    html.Td(int(row["RecipeID"])),
                    html.Td(
                        smiles,
                        style={"fontSize": "11px", "wordBreak": "break-all"},
                    ),
                    html.Td(f"{float(row['LD50']):.4f}"),
                    cell,
                ]
            )
        )

    table = html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={"width": "100%", "borderCollapse": "collapse"},
    )
    return table


@app.callback(
    Output("structure-modal", "style"),
    Output("structure-modal-img", "src"),
    Input({"type": "top-ten-structure", "smiles": ALL}, "n_clicks"),
    Input("structure-modal-close", "n_clicks"),
    State("structure-modal", "style"),
    prevent_initial_call=True,
)
def handle_structure_click(_structure_clicks, _close_clicks, current_style):
    """Open the modal with the clicked structure, or close it."""
    triggered = callback_context.triggered
    if not triggered:
        raise PreventUpdate

    trigger_id = triggered[0]["prop_id"].split(".")[0]

    hidden = {**(current_style or {}), "display": "none"}
    shown = {**(current_style or {}), "display": "flex"}

    # Close button.
    if trigger_id == "structure-modal-close":
        return hidden, None

    # An image was clicked. Its id is a JSON string carrying the SMILES.
    try:
        parsed = json.loads(trigger_id)
        smiles = parsed["smiles"]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise PreventUpdate

    # Ignore the initial n_clicks=0 fire.
    if triggered[0]["value"] in (None, 0):
        raise PreventUpdate

    uri = _smiles_to_png_uri(smiles, MODAL_SIZE)
    if uri is None:
        raise PreventUpdate

    return shown, uri
