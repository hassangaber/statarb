#!/usr/bin/env/ python3.11
import pandas as pd
import dash

from dash import Input, Output

from api.web_helpers.layout import export_layout
from api.web_helpers.callbacks import register_callbacks
from api.web_helpers.pages import ( render_intro, render_montecarlo, 
                                    render_analyze, render_rf, 
                                    render_backtest_ml, render_theory,
                                    render_hmm)



df = pd.read_csv("assets/data.csv")
df.DATE = pd.to_datetime(df.DATE)
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = export_layout(df)

@app.callback(Output("page-content", "children"), 
              [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/":
        return render_intro()
    elif pathname == "/analyze":
        return render_analyze(df)
    elif pathname == "/montecarlo":
        return render_montecarlo(df)
    elif pathname == "/backtest-ml":
        return render_backtest_ml(df)
    elif pathname == "/theory":
        return render_theory()
    elif pathname == '/backtest-rf':
        return render_rf(df)
    elif pathname == '/markov':
        return render_hmm(df)
    else:
        return "404 - Page not found"

register_callbacks(app, df)

if __name__ == "__main__":
    app.run_server(debug=True)
