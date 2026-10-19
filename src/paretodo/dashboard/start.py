"""Dash app for visualizing the Pareto front from MLflow experiments."""

from paretodo.dashboard import (
    app,
    get_app_layout,
)

app.title = "Pareto Front Visualization (Version 2)"
app.layout = get_app_layout()


if __name__ == "__main__":
    app.run(debug=True, port=8055)
