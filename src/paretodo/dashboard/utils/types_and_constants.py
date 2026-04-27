"""Type definitions and constants for the Pareto visualization app."""

from typing import TypeAlias

PointIndex: TypeAlias = int
ClickedPointsData: TypeAlias = dict[str, list[PointIndex]]
RenamesDict: TypeAlias = dict[str, str]
BoundsDict: TypeAlias = dict[str, dict[str, float | None]]
ObjectiveBoundsData: TypeAlias = dict[str, dict[str, float | None]]

LOCAL_TEMP_DIR = ".temp/"

TABLE_HEADER_STYLE = {
    "border": "2px solid #2c3e50",
    "padding": "12px",
    "textAlign": "left",
    "backgroundColor": "#34495e",
    "color": "white",
    "fontWeight": "bold",
}

TABLE_CELL_STYLE = {
    "border": "1px solid #bdc3c7",
    "padding": "12px",
    "fontWeight": "500",
}

TABLE_ROW_ALT_BG = "#ecf0f1"
TABLE_ROW_NORMAL_BG = "white"

MARKER_COLORS = {
    "pareto": "#648FFF",
    "selected": "#FFB000",
    "experiment": "#E885B4",
    "highlight": "#FF001E",
    "auto_selected": "#FFA500",
    "discarded": "black",
}

MARKER_SYMBOLS = {
    "pareto": "circle",
    "experiment": "square",
    "selected": "x",
    "highlight": "cross",
    "auto_selected": "diamond",
}

MARKER_NAMES = {
    "pareto": "Pareto Front",
    "selected": "Sel. Recipe",
    "experiment": "Experiments",
    "highlight": "Sel. Exp.",
    "auto_selected": "Auto Sel. Rec.",
    "discarded": "Discarded",
}
