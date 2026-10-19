"""Callbacks for browser viewport tracking."""

from dash import Input, Output

from ..server import app

__all__: list[str] = []


app.clientside_callback(
    """
    function(nIntervals) {
        const width = window.innerWidth || 0;
        if (window.__pfgsLastViewportWidth === width) {
            return window.dash_clientside.no_update;
        }
        window.__pfgsLastViewportWidth = width;
        return width;
    }
    """,
    Output("viewport-width-store", "data"),
    Input("viewport-width-poller", "n_intervals"),
    prevent_initial_call=True,
)
