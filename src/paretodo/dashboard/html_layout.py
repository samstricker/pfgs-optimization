"""HTML Layout for the Pareto Front Visualization Dash App (Version 2 - Minimal)."""

from dash import html, dcc


def get_app_layout() -> html.Div:
    """Create the HTML layout for the Pareto front visualization Dash app (v2 - minimal).

    Returns
    -------
    html.Div
        The HTML layout as a Dash html.Div object.
    """
    layout = html.Div(
        [
            # Store for tracking clicked points
            dcc.Store(
                id="clicked-points-store",
                data={
                    "investigation-space": [],
                    "pareto-front": [],
                    "experiment-results": [],
                },
            ),
            # Store for objectives bounds
            dcc.Store(
                id="objectives-bounds-store",
                data={},
            ),
            # Store for pinned pareto points
            dcc.Store(
                id="pinned-points-store",
                data={},
            ),
            # Store for objective renames
            dcc.Store(
                id="objective-renames-store",
                data={},
            ),
            # Store for investigation space variable renames
            dcc.Store(
                id="investigation-space-renames-store",
                data={},
            ),
            # Store for measurement renames
            dcc.Store(
                id="measurement-renames-store",
                data={},
            ),
            # Store for experiment result figures
            dcc.Store(
                id="experiment-results-figures-store",
                data=[],
            ),
            # Store for GP prediction figures
            dcc.Store(
                id="gp-predictions-figures-store",
                data=[],
            ),
            # Download component for exporting plots and tables
            dcc.Download(id="download-plots-tables"),
            html.H1("Pareto Front Visualization (Version 2)"),
            html.Hr(style={"border": "2px solid #333", "margin": "20px 0"}),
            html.Div(
                [
                    # Left column: Controls
                    html.Div(
                        [
                            html.H2("Settings"),
                            html.Div(style={"height": "10px"}),
                            html.Div(
                                [
                                    html.Label("MLflow Port:"),
                                    dcc.Input(
                                        id="mlflow-port",
                                        type="number",
                                        value=5000,
                                        style={"width": "100%", "marginTop": "5px"},
                                    ),
                                ],
                                style={"marginBottom": "15px"},
                            ),
                            html.Div(
                                [
                                    html.Label("MLflow Experiment:"),
                                    dcc.Dropdown(
                                        id="mlflow-experiment",
                                        options=[],
                                        value=None,
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"marginBottom": "15px"},
                            ),
                            html.Div(
                                [
                                    html.Label("MLflow Run:"),
                                    dcc.Dropdown(
                                        id="mlflow-run",
                                        options=[],
                                        value=None,
                                        style={"width": "100%"},
                                    ),
                                    html.Button(
                                        "Refresh Runs",
                                        id="refresh-runs-button",
                                        n_clicks=0,
                                        style={
                                            "width": "100%",
                                            "marginTop": "5px",
                                            "padding": "8px",
                                        },
                                    ),
                                ],
                                style={"marginBottom": "15px"},
                            ),
                            html.Div(
                                [
                                    html.Label("Iteration:"),
                                    dcc.Dropdown(
                                        id="iteration",
                                        options=[],
                                        value=None,
                                        style={"width": "100%"},
                                    ),
                                ],
                                style={"marginBottom": "15px"},
                            ),
                            html.Button(
                                "Download Plots & Tables",
                                id="download-button",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "marginTop": "5px",
                                    "padding": "8px",
                                    "backgroundColor": "#27ae60",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                    "fontWeight": "bold",
                                },
                            ),
                        ],
                        style={
                            "width": "25%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "padding": "20px",
                            "backgroundColor": "#f5f5f5",
                            "borderRadius": "5px",
                        },
                    ),
                    # Right column: Data tables
                    html.Div(
                        [
                            html.H2("Investigation Space"),
                            html.Div(
                                id="investigation-space-table",
                                style={"marginBottom": "30px"},
                            ),
                            html.H2("Optimization Problem"),
                            html.Div(
                                id="optimization-problem-table",
                                style={"marginBottom": "30px"},
                            ),
                        ],
                        style={
                            "width": "72%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "padding": "20px",
                            "marginLeft": "2%",
                        },
                    ),
                ],
                style={"display": "flex", "width": "100%"},
            ),
            # Experiment Results Visualization Section
            html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
            html.H2("Experiment Results"),
            html.Div(
                id="experiment-results-container",
                style={"width": "100%"},
            ),
            # GP Predictions Section
            html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
            html.H2("GP Predictions"),
            html.Div(
                id="gp-filters-container",
                style={"marginBottom": "20px"},
            ),
            html.Div(
                id="gp-predictions-container",
                style={"width": "100%"},
            ),
            # Investigation Space Visualization Section
            html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
            html.H2("Investigation Space Visualization"),
            dcc.Graph(
                id="investigation-space-graph",
                style={"width": "100%"},
            ),
            # Pareto Front Visualization Section
            html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
            html.H2("Pareto Front Visualization"),
            html.Div(
                id="objectives-table-container",
                style={"marginBottom": "20px"},
            ),
            dcc.Graph(
                id="pareto-front-graph",
                style={"width": "100%"},
            ),
            # Selection Controls Section
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Batch Size:",
                                style={"marginRight": "10px", "fontWeight": "bold"},
                            ),
                            dcc.Input(
                                id="batch-size-input",
                                type="number",
                                value=5,
                                min=1,
                                style={
                                    "padding": "8px 8px",
                                    "border": "1px solid #bdc3c7",
                                    "borderRadius": "4px",
                                    "width": "120px",
                                    "height": "44px",
                                    "lineHeight": "28px",
                                    "marginRight": "20px",
                                    "fontSize": "14px",
                                },
                            ),
                            html.Button(
                                "Automated Selection",
                                id="automated-selection-button",
                                n_clicks=0,
                                style={
                                    "padding": "8px 16px",
                                    "backgroundColor": "#3498db",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                    "fontWeight": "bold",
                                    "marginRight": "10px",
                                },
                            ),
                            html.Button(
                                "Reset",
                                id="reset-selection-button",
                                n_clicks=0,
                                style={
                                    "padding": "8px 16px",
                                    "backgroundColor": "#e74c3c",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                    "fontWeight": "bold",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginTop": "20px",
                            "marginBottom": "20px",
                            "padding": "15px",
                            "backgroundColor": "#f5f5f5",
                            "borderRadius": "4px",
                        },
                    ),
                ],
                style={"width": "100%"},
            ),
            # Pareto Point Details Section
            html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
            html.H2("Pareto Point Details"),
            html.Div(
                id="pareto-details-table-container",
                style={"width": "100%", "overflowX": "auto"},
            ),
        ],
        style={
            "fontFamily": "Arial, sans-serif",
            "padding": "20px",
            "width": "100vw",
            "boxSizing": "border-box",
            "margin": "0",
            "overflowX": "hidden",
        },
    )
    return layout
