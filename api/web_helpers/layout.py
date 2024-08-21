from dash import html, dcc
import pandas as pd

def export_layout() -> html.Div:
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            html.Div(
                [
                    html.Div(
                        [
                            html.H6("Contents", className="sidebar-header"),
                            html.Ul(
                                [
                                    html.Li(html.A("Introduction", href="/")),
                                    html.Li(html.A("1 Risk Models", href="/risk")),
                                    html.Li(html.A("2 Equity Signals I", href="/eq-one")),
                                ],
                                className="sidebar-list",
                            ),
                        ],
                        className="sidebar",
                    ),

                    html.Div(id="page-content", className="content"),
                ],
                className="container-fluid",
            ),
        ]
    )
