#!/usr/bin/env/ python3.11
import dash
import dash_bootstrap_components as dbc  # type: ignore
import pandas as pd
from dash import Input, Output

from api.web_helpers.dash_call_you_back import register_callbacks
from api.web_helpers.layout import export_layout
from api.web_helpers.pages import (render_analyze, render_combined_page_ml,
                                   render_hmm, render_intro, render_montecarlo_combied)

global APP

df = pd.read_csv("assets/data.csv")
df.DATE = pd.to_datetime(df.DATE)

APP = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

server = APP.server

APP.layout = export_layout(df)


@APP.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/":
        return render_intro()
    elif pathname == "/analyze":
        return render_analyze(df)
    elif pathname == "/montecarlo":
        return render_montecarlo_combied(df)
    elif pathname == "/stochastic-signals":
        return render_combined_page_ml(df)
    elif pathname == "/markov":
        return render_hmm(df)
    else:
        return "404 - Page not found"


register_callbacks(APP, df)

if __name__ == "__main__":
    APP.run(debug=True)
