"""Render the clicked Pareto point's molecule with RDKit and display its SMILES string and recipe ID."""

import base64
from io import BytesIO

from dash import Input, Output, html
from dash.exceptions import PreventUpdate

from ..server import app

__all__ = ["update_molecule_preview"]


def _smiles_to_data_uri(smiles: str) -> str | None:
    """Render a SMILES string to a base64 PNG data URI, or None if invalid"""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    drawer = rdMolDraw2D.MolDraw2DCairo(320, 260)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()

    encoded = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.callback(
    Output("molecule-preview-img", "src"),
    Output("molecule-preview-caption", "children"),
    Input("pareto-front-graph", "clickData"),
)
def update_molecule_preview(click_data):
    """Draw the clicked molecule. customdata[2] is the SMILES string"""
    if not click_data:
        raise PreventUpdate

    try:
        point = click_data["points"][0]
        customdata = point["customdata"]
        recipe_id = customdata[1]
        smiles = str(customdata[2])
    except (KeyError, IndexError, TypeError):
        raise PreventUpdate

    uri = _smiles_to_data_uri(smiles)
    if uri is None:
        return None, html.Div(f"Could not render SMILES: {smiles}")

    caption = html.Div(
        [
            html.Div(f"RecipeID: {int(float(recipe_id))}"),
            html.Div(smiles, style={"fontSize": "11px", "wordBreak": "break-all"}),
        ]
    )
    return uri, caption