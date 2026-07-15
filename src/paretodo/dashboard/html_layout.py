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
            # Store for alternative expected error values
            dcc.Store(
                id="alternative-expected-error-store",
                data={},
            ),
            # Store for pareto mode (Optimization or Manufacturing)
            dcc.Store(
                id="pareto-mode-store",
                data="Optimization",
            ),
            # Store for viewport width so plots can adapt to screen size
            dcc.Store(
                id="viewport-width-store",
                data=0,
            ),
            # Polls the browser width so the Pareto plot can react to resizing
            dcc.Interval(
                id="viewport-width-poller",
                interval=1000,
                n_intervals=0,
            ),
            # Download component for exporting plots and tables
            dcc.Download(id="download-plots-tables"),
            # Structure modal (overlay for clicked molecule)
            html.Div(
                id="structure-modal",
                children=[
                    html.Div(
                        [
                            html.Button(
                                "Close",
                                id="structure-modal-close",
                                n_clicks=0,
                                style={"float": "right"},
                            ),
                            html.Img(id="structure-modal-img"),
                        ],
                        style={
                            "background": "white",
                            "padding": "20px",
                            "borderRadius": "8px",
                        },
                    )
                ],
                style={
                    "display": "none",
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "width": "100%",
                    "height": "100%",
                    "background": "rgba(0,0,0,0.5)",
                    "zIndex": "1000",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            ),
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
                                    html.Button(
                                        "Refresh Iterations",
                                        id="refresh-iterations-button",
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
                                    html.Label("Pareto Mode:"),
                                    dcc.Dropdown(
                                        id="pareto-mode",
                                        options=[
                                            {
                                                "label": "Optimization",
                                                "value": "Optimization",
                                            },
                                            {
                                                "label": "Manufacturing",
                                                "value": "Manufacturing",
                                            },
                                        ],
                                        value="Optimization",
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
            html.Div(
                id="experiment-results-section",
                children=[
                    html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
                    html.H2("Experiment Results"),
                    html.Div(
                        id="experiment-results-container",
                        style={"width": "100%"},
                    ),
                ],
            ),
            # Experiment Results Parallel Coordinates Section
            html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
            html.H2("Experiment Results - Parallel Coordinates"),
            dcc.Graph(
                id="experiment-results-parcoords-graph",
                style={"width": "100%"},
            ),
            # Top 10 Observed Molecules Section
            html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
            html.H2("Top 10 Observed Molecules"),
            html.Div(id="top-ten-table-container"),
            # GP Predictions Section
            html.Div(
                id="gp-predictions-section",
                children=[
                    html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
                    html.H2("GP Predictions"),
            html.Div(
                id="gp-filters-container",
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="gp-show-reference-point",
                        options=[
                            {
                                "label": " Show Reference Point",
                                "value": "show_ref",
                            },
                        ],
                        value=["show_ref"],
                        style={"display": "flex"},
                        labelStyle={"marginRight": "10px"},
                    ),
                ],
                style={
                    "padding": "10px",
                    "backgroundColor": "#f5f5f5",
                    "borderRadius": "5px",
                    "marginBottom": "15px",
                },
            ),
            # Robustness Section
            html.Div(
                [
                    html.H3("Robustness", style={"marginBottom": "15px"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        "Confidence Level (%):",
                                        style={
                                            "marginRight": "10px",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="robustness-confidence-level",
                                        options=[
                                            {"label": f"{val}%", "value": val}
                                            for val in [25, 50, 75, 80, 85, 90, 95, 99]
                                        ],
                                        value=95,
                                        style={"width": "150px"},
                                    ),
                                ],
                                style={
                                    "display": "inline-block",
                                    "marginRight": "30px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Pareto Point:",
                                        style={
                                            "marginRight": "10px",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="robustness-pareto-point",
                                        options=[],
                                        value=None,
                                        style={"width": "250px"},
                                    ),
                                ],
                                style={"display": "inline-block"},
                            ),
                        ],
                        style={
                            "padding": "10px",
                            "backgroundColor": "#f5f5f5",
                            "borderRadius": "5px",
                            "marginBottom": "15px",
                        },
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(
                id="gp-predictions-container",
                style={"width": "100%"},
            ),
                ],
            ),
            # Investigation Space Visualization Section
            html.Div(
                id="investigation-space-viz-section",
                children=[
                    html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
                    html.H2("Investigation Space Visualization"),
                    dcc.Graph(
                        id="investigation-space-graph",
                        style={"width": "100%"},
                    ),
                ],
            ),
            # Pareto Front Visualization Section
            html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
            html.H2("Pareto Front Visualization"),
            html.Div(
                id="objectives-table-container",
                style={"marginBottom": "20px"},
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
            html.Div(
                [
                    dcc.Graph(
                        id="pareto-front-graph",
                        style={"width": "70%", "display": "inline-block"},
                    ),
                    html.Div(
                        [
                            html.H4("Selected molecule"),
                            html.Img(
                                id="molecule-preview-img",
                                style={"maxWidth": "100%"},
                            ),
                            html.Div(id="molecule-preview-caption"),
                        ],
                        style={
                            "width": "28%",
                            "display": "inline-block",
                            "verticalAlign": "top",
                            "paddingLeft": "2%",
                        },
                    ),
                ],
                style={"width": "100%"},
            ),
            # Pareto Front Parallel Coordinates Section
            html.Hr(style={"border": "2px solid #333", "margin": "30px 0"}),
            html.H2("Pareto Front - Parallel Coordinates"),
            dcc.Graph(
                id="pareto-front-parcoords-graph",
                style={"width": "100%"},
            ),
            # Pareto Parallel Coordinates Controls
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Columns to Display:",
                                style={
                                    "fontWeight": "bold",
                                    "marginBottom": "10px",
                                    "display": "block",
                                },
                            ),
                            dcc.Checklist(
                                id="parcoords-columns-checklist",
                                options=[],  # Will be populated by callback
                                value=[],  # Will be populated by callback
                                style={
                                    "display": "flex",
                                    "flexWrap": "wrap",
                                    "gap": "15px",
                                },
                                labelStyle={"marginRight": "10px"},
                            ),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                    html.Div(
                        [
                            dcc.Checklist(
                                id="parcoords-visibility-checklist",
                                options=[
                                    {
                                        "label": " Show Experiments",
                                        "value": "show_experiments",
                                    },
                                    {
                                        "label": " Show Selected Recipes",
                                        "value": "show_selected",
                                    },
                                ],
                                value=["show_experiments", "show_selected"],
                                style={"display": "flex", "gap": "30px"},
                                labelStyle={"marginRight": "10px"},
                            ),
                        ],
                        style={"marginBottom": "20px"},
                    ),
                ],
                style={
                    "padding": "15px",
                    "backgroundColor": "#f8f9fa",
                    "border": "1px solid #dee2e6",
                    "borderRadius": "4px",
                    "marginBottom": "20px",
                },
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
