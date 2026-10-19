"""Server setup for the Pareto front visualization app."""

from dash import Dash
from flask import Flask

server = Flask("myproject")
app = Dash(server=server, suppress_callback_exceptions=True)
