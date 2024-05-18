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
                            html.Li(html.A("1 Analyze Time Series", href="/analyze")),
                            html.Li(html.A("2 Monte-Carlo Portfolio Simulation", href="/montecarlo")),
                            html.Li(html.A("3.1 Backtest with NN Approach", href="/backtest-ml")),
                            html.Li(html.A("3.2 Labeling Theory", href="/theory")),
                            html.Li(html.A("3.3 Backtest with RF Approach", href='/backtest-rf')),
                            html.Li(html.A("4 Hidden Markov Models with ML Signals", href='/markov')),
                            #html.Li(html.A("4 Backtest with Indicators", href="/backtest-indicators")),
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
                    "padding": "15px",
                    "background-color": "#f8f9fa",
                },
            ),
            html.Div(
                id="page-content",
                style={"margin-left": "220px", "padding": "20px"}
            ),
        ]
    )