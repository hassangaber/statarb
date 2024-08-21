
import pandas as pd


import dash
from dash import callback_context, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

from api.web_helpers.callback.risk_intro import risk_intro



def register_callbacks(app: dash.Dash, df: pd.DataFrame) -> None:

    @app.callback(
        [
            Output("volatility-output", "children"),
            Output("price-volatility-plot", "figure"),
            Output("long-position-plot", "figure"),
            Output("short-position-plot", "figure"),
            Output("trade-risk-profile", "children")
        ],
        [Input("calc-volatility", "n_clicks")],
        [State("ticker-input", "value")]
    )
    def update_volatility_analysis(n_clicks, ticker):
        if n_clicks is None or ticker is None:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # Check if the ticker exists in the dataframe
        if ticker not in df['ID'].values:
            empty_fig = go.Figure()
            empty_div = html.Div("No data available")
            return empty_div, empty_fig, empty_fig, empty_fig, empty_div
        
        ticker_data = df.loc[df.ID == ticker]
        
        # Check if we have data for the selected ticker
        if ticker_data.empty:
            empty_fig = go.Figure()
            empty_div = html.Div("No data available for the selected ticker")
            return empty_div, empty_fig, empty_fig, empty_fig, empty_div

        # If we have data, proceed with the risk analysis
        try:
            res = risk_intro(data=ticker_data, t=ticker)
            return res
        except Exception as e:
            error_message = html.Div(f"An error occurred: {str(e)}")
            empty_fig = go.Figure()
            return error_message, empty_fig, empty_fig, empty_fig, error_message

