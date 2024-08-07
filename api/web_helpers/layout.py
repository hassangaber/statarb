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
                            html.Li(html.A("1 View Time-Series", href="/analyze")),
                            html.Li(html.A("2 Risk Forecast", href="/montecarlo")),
                            html.Li(html.A("3 Trading Signals", href="/stochastic-signals")),
                            html.Li(html.A("4 Mixed Trading Signals", href="/markov")),
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
                    "width": "225px",
                    "padding": "10px",
                    "background-color": "#f8f9fa",
                },
            ),
            html.Div(id="page-content", style={"margin-left": "220px", "padding": "20px"}),
        ]
    )
