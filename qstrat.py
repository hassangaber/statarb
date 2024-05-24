#!/usr/bin/env/ python3.11
import pandas as pd
import dash

from dash import Input, Output
import dash_bootstrap_components as dbc

from api.web_helpers.layout import export_layout
from api.web_helpers.callbacks import register_callbacks
from api.web_helpers.pages import (
    render_intro,
    render_montecarlo,
    render_analyze,
    render_combined_page,
    render_hmm,
)


df = pd.read_csv("assets/data.csv")
df.DATE = pd.to_datetime(df.DATE)
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = export_layout(df)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/":
        return render_intro()
    elif pathname == "/analyze":
        return render_analyze(df)
    elif pathname == "/montecarlo":
        return render_montecarlo(df)
    elif pathname == "/quant-signals":
        return render_combined_page(df)
    elif pathname == "/markov":
        return render_hmm(df)
    else:
        return "404 - Page not found"


register_callbacks(app, df)

if __name__ == "__main__":
    app.run(debug=True)
