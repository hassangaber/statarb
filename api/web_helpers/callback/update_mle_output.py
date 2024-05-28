import urllib
from typing import Any

from dash import dash_table
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import html
from scipy.stats import gaussian_kde


def update_mle_output(data_json: str) -> tuple[Any, Any, Any, Any, str, Any, Any, Any]:
    if not data_json:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data to display", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
        return empty_fig, empty_fig, empty_fig, empty_fig, "", "", empty_fig, ""

    action_df = pd.read_json(data_json, orient="split")
    action_df.DATE = pd.to_datetime(action_df.DATE)

    graph_layout = go.Layout(
        title="",
        xaxis=dict(title="Date", showgrid=True, zeroline=False),
        yaxis=dict(title="", showgrid=True, zeroline=False),
        hovermode="closest",
        template="plotly_white",
    )

    portfolio_value_fig = go.Figure(
        data=[
            go.Scatter(
                x=action_df.DATE,
                y=action_df["total_portfolio_value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(shape="spline", smoothing=1.3),
            ),
        ],
        layout=graph_layout.update(title="Portfolio Value Over Time", yaxis_title="Value ($)"),
    )

    cash_on_hand_fig = go.Figure(
        data=[
            go.Scatter(
                x=action_df.DATE,
                y=action_df["cash_on_hand"],
                mode="lines",
                name="Cash on Hand",
                line=dict(shape="spline", smoothing=1.3),
            ),
        ],
        layout=graph_layout.update(title="Cash on Hand Over Time", yaxis_title="Value ($)"),
    )

    pnl_fig = go.Figure(
        data=[
            go.Scatter(
                x=action_df.DATE,
                y=action_df["PnL"],
                mode="lines",
                name="Profit and Loss",
                line=dict(shape="linear", smoothing=1.3),
            ),
            go.Scatter(
                x=action_df.DATE,
                y=[0] * len(action_df),
                mode="lines",
                name="Break-Even Level",
                line=dict(color="red", dash="dash"),
            ),
        ],
        layout=graph_layout.update(title="PnL Over Time", yaxis_title="Value ($)"),
    )

    transaction_signals_fig = go.Figure(
        data=[
            go.Scatter(
                x=action_df.DATE,
                y=action_df["CLOSE"],
                mode="lines",
                name="Close Price",
                line=dict(shape="linear", smoothing=1.3),
            ),
            go.Scatter(
                x=action_df[action_df["graph_signal"] == 2].DATE,
                y=action_df[action_df["graph_signal"] == 2]["CLOSE"],
                mode="markers",
                marker=dict(color="green", size=10, symbol="triangle-up"),
                name="Buy Signal",
            ),
            go.Scatter(
                x=action_df[(action_df["graph_signal"] == 0) & (action_df["cumulative_shares"] > 0)].DATE,
                y=action_df[(action_df["graph_signal"] == 0) & (action_df["cumulative_shares"] > 0)]["CLOSE"],
                mode="markers",
                marker=dict(color="red", size=10, symbol="triangle-down"),
                name="Sell Signal",
            ),
        ],
        layout=graph_layout.update(title="Transaction Signals Over Time", yaxis_title="Close Price (US$)"),
    )

    csv_string = action_df.reset_index().to_csv(index=False, encoding="utf-8")
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

    def format_floats(df: pd.DataFrame) -> pd.DataFrame:
        float_cols = df.select_dtypes(include=["float"]).columns
        for col in float_cols:
            df[col] = df[col].map("{:.4f}".format)
        return df

    formatted_df = format_floats(action_df.copy())
    formatted_df = formatted_df.drop(columns=["predicted_signal"])

    data_table = dash_table.DataTable(
        data=formatted_df.reset_index().to_dict("records"),
        columns=[{"name": i, "id": i, "deletable": False, "renamable": False, "editable": False} for i in formatted_df.columns],
        style_table={"overflowX": "auto", "maxHeight": "500px", "overflowY": "auto"},
        style_cell={"padding": "5px", "whiteSpace": "normal", "height": "auto", "fontSize": 12},
        style_header={"fontWeight": "bold", "fontSize": 14},
        export_format="csv",
        export_headers="display",
    )

    returns = action_df["RETURNS"].dropna()
    kde = gaussian_kde(returns)
    x = np.linspace(returns.min(), returns.max(), 1000)
    y = kde(x)

    returns_distribution_fig = go.Figure(
        data=[
            go.Histogram(x=returns, nbinsx=150, name="Returns Histogram", opacity=0.5, histnorm="probability density"),
            go.Scatter(x=x, y=y, mode="lines", name="Returns KDE", line=dict(shape="spline", smoothing=1.3)),
        ],
        layout=graph_layout.update(title="KDE and Histogram of Returns", xaxis_title="Returns", yaxis_title="Density"),
    )
    returns_distribution_fig.update_layout(height=300)

    mean_return = returns.mean()
    std_return = returns.std()

    stats_container = html.Div(
        [
            html.P(f"Mean Return: {mean_return:.4f}", style={"font-size": "18px"}),
            html.P(f"Standard Deviation of Returns: {std_return:.4f}", style={"font-size": "18px"}),
        ]
    )

    return (
        portfolio_value_fig,
        transaction_signals_fig,
        cash_on_hand_fig,
        pnl_fig,
        data_table,
        csv_string,
        returns_distribution_fig,
        stats_container,
    )
