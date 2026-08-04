"""Render the hovered or clicked Pareto point's molecule with RDKit."""

import base64

from dash import Input, Output, html
from dash.exceptions import PreventUpdate

from ..server import app

__all__ = ["update_molecule_preview"]


def _smiles_to_data_uri(smiles: str) -> str | None:
    """Render a SMILES string to a base64 PNG data URI, or None if invalid."""
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
    Input("pareto-front-graph", "hoverData"),
    Input("pareto-front-graph", "clickData"),
)
def update_molecule_preview(hover_data, click_data):
    """
    Draw the hovered molecule temporarily.

    When the cursor leaves the point, display the most recently clicked
    molecule instead. customdata[1] is the recipe ID and customdata[2]
    is the SMILES string.
    """
    # Hovered points take priority over the last clicked point.
    point_data = hover_data if hover_data else click_data

    if not point_data:
        raise PreventUpdate

    try:
        point = point_data["points"][0]
        customdata = point["customdata"]
        recipe_id = customdata[1]
        smiles = str(customdata[2])
    except (KeyError, IndexError, TypeError):
        raise PreventUpdate

    uri = _smiles_to_data_uri(smiles)

    if uri is None:
        return None, html.Div(f"Could not render SMILES: {smiles}")

    caption_style = {
        "fontSize": "16px",
        "lineHeight": "1.4",
        "marginBottom": "4px",
    }

    caption = html.Div(
        [
            html.Div(
                [
                    html.Strong("Recipe ID: "),
                    str(int(float(recipe_id))),
                ],
                style=caption_style,
            ),
            html.Div(
                [
                    html.Strong("SMILES: "),
                    smiles,
                ],
                style={
                    **caption_style,
                    "wordBreak": "break-all",
                },
            ),
        ]
    )

    return uri, caption