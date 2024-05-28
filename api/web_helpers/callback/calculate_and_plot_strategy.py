import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from api.src.compute import compute_signal


def calculate_and_plot_strategy(df: pd.DataFrame, stock_id: str, start_date: str, end_date: str) -> go.Figure:
    M = df.loc[df.ID == stock_id]
    M = M[(M.DATE >= pd.to_datetime(start_date)) & (M.DATE <= pd.to_datetime(end_date))]
    M = compute_signal(M)

    M["system_returns"] = M["RETURNS"] * M["signal"]
    M["entry"] = M.signal.diff()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Price and Signals", "Cumulative Returns"),
    )

    fig.add_trace(
        go.Scatter(x=M["DATE"], y=M["CLOSE"], mode="lines", name="Close Price"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=M["DATE"], y=M["CLOSE_SMA_9D"], mode="lines", name="9-day SMA"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=M["DATE"], y=M["CLOSE_SMA_21D"], mode="lines", name="21-day SMA"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=M["DATE"][M["entry"] == 2],
            y=M["CLOSE"][M["entry"] == 2],
            mode="markers",
            marker=dict(color="green", size=15, symbol="triangle-up"),
            name="Long Entry",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=M["DATE"][M["entry"] == -2],
            y=M["CLOSE"][M["entry"] == -2],
            mode="markers",
            marker=dict(color="red", size=15, symbol="triangle-down"),
            name="Short Entry",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=M["DATE"],
            y=np.exp(M["RETURNS"]).cumprod(),
            mode="lines",
            name="Buy/Hold",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=M["DATE"],
            y=np.exp(M["system_returns"]).cumprod(),
            mode="lines",
            name="Strategy",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=1000,
        width=2000,
        title_text=f"Trading Strategy Analysis for {stock_id}",
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Returns", row=2, col=1)

    return fig
