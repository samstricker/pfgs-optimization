"""Callbacks for browser viewport tracking."""

from dash import Input, Output

from ..server import app

__all__: list[str] = []


app.clientside_callback(
    """
    function(nIntervals) {
        return window.innerWidth || 0;
    }
    """,
    Output("viewport-width-store", "data"),
    Input("viewport-width-poller", "n_intervals"),
    prevent_initial_call=True,
)
