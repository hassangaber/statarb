import pandas as pd
from dash import dcc, html

def export_layout(df: pd.DataFrame) -> html.Div:
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            html.Div(
                [
                    html.H2("Table of Contents", className="sidebar-header"),
                    html.Ul(
                        [
                            html.Li(html.A("Introduction", href="/")),
                            html.Li(html.A("Analyze Time Series", href="/analyze")),
                            html.Li(html.A("Monte-Carlo Portfolio Simulation", href="/montecarlo")),
                            html.Li(html.A("Backtest with ML Models", href="/backtest-ml")),
                            html.Li(html.A("Theory", href="/theory")),
                            html.Li(html.A("Backtest with Indicators", href="/backtest-indicators")),
                        ],
                        className="sidebar-list",
                    ),
                ],
                className="sidebar",
                style={
                    "position": "fixed",
                    "top": "0",
                    "left": "0",
                    "bottom": "0",
                    "width": "200px",
                    "padding": "20px",
                    "background-color": "#f8f9fa",
                },
            ),
            html.Div(
                id="page-content",
                style={"margin-left": "220px", "padding": "20px"}
            ),
        ]
    )