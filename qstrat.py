#!/usr/bin/env/ python3.11
import dash
import dash_bootstrap_components as dbc  # type: ignore
import pandas as pd
from dash import Input, Output

from api.web_helpers.callbacks import register_callbacks
from api.web_helpers.layout import export_layout
from api.web_helpers.pages import render_eq1, render_intro, render_strat1

global APP

df = pd.read_csv("assets/data.csv")
df.DATE = pd.to_datetime(df.DATE)

APP = dash.Dash(__name__, 
                suppress_callback_exceptions=True, 
                external_stylesheets=[dbc.themes.BOOTSTRAP]
            )

server = APP.server

APP.layout = export_layout()


@APP.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname) -> callable:
    if pathname == "/":
        return render_intro()
    elif pathname == "/equity-1":
        return render_strat1()
    elif pathname == "/equity-2":
        return 
    
    else:
        return "404 - Page not found"


register_callbacks(APP, df)

if __name__ == "__main__":
    APP.run(debug=True)
