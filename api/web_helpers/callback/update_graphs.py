import numpy as np
import pandas as pd
import plotly.graph_objs as go

from api.web_helpers.utils import kde_scipy, random_color


def filter_df(df: pd.DataFrame, selected_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    filtered_df = df[df["ID"] == selected_id]
    filtered_df = filtered_df[(filtered_df["DATE"] >= start_date) & (filtered_df["DATE"] <= end_date)]
    return filtered_df


def create_main_traces(
    filtered_df: pd.DataFrame, selected_id: str, selected_data: list[str], selected_days: list[int]
) -> list[go.Scatter | go.Bar]:
    traces_main = []
    if "CLOSE" in selected_data:
        traces_main.append(
            go.Scatter(
                x=filtered_df["DATE"],
                y=filtered_df["CLOSE"].astype(float),
                mode="lines",
                name=f"{selected_id} Close",
                line=dict(color=random_color()),
                yaxis="y1",
            )
        )
        traces_main.append(
            go.Bar(
                x=filtered_df["DATE"],
                y=filtered_df["VOLUME"],
                name=f"{selected_id} Volume",
                marker=dict(color=random_color()),
                opacity=0.6,
                yaxis="y2",
            )
        )

    for day in selected_days:
        if "SMA" in selected_data:
            sma_column = f"CLOSE_SMA_{day}D"
            if sma_column in filtered_df.columns:
                traces_main.append(
                    go.Scatter(
                        x=filtered_df["DATE"],
                        y=filtered_df[sma_column],
                        mode="lines",
                        name=f"{selected_id} SMA {day}D",
                        line=dict(color=random_color()),
                    )
                )

        if "EWMA" in selected_data:
            ewma_column = f"CLOSE_EWMA_{day}D"
            if ewma_column in filtered_df.columns:
                traces_main.append(
                    go.Scatter(
                        x=filtered_df["DATE"],
                        y=filtered_df[ewma_column],
                        mode="lines",
                        name=f"{selected_id} EWMA {day}D",
                        line=dict(color=random_color()),
                    )
                )
    return traces_main


def create_returns_traces(filtered_df: pd.DataFrame, selected_id: str) -> list[go.Histogram | go.Scatter]:
    traces_returns = []
    if True:
        total_counts = len(filtered_df["RETURNS"].dropna())
        bin_size = (np.max(filtered_df["RETURNS"]) - np.min(filtered_df["RETURNS"])) / 40
        histogram_scaling_factor = total_counts * bin_size

        traces_returns.append(
            go.Histogram(
                x=filtered_df["RETURNS"],
                name=f"{selected_id} Returns Frequency",
                marker=dict(color=random_color()),
                opacity=0.75,
                xbins=dict(
                    start=np.min(filtered_df["RETURNS"]),
                    end=np.max(filtered_df["RETURNS"]),
                    size=bin_size,
                ),
                autobinx=False,
            )
        )

        x_grid = np.linspace(np.min(filtered_df["RETURNS"]), np.max(filtered_df["RETURNS"]), 1000)
        pdf = kde_scipy(filtered_df["RETURNS"].dropna(), x_grid, bandwidth=0.2)

        traces_returns.append(
            go.Scatter(
                x=x_grid,
                y=pdf * histogram_scaling_factor,
                mode="lines",
                name=f"{selected_id} Returns KDE",
                line=dict(color=random_color(), width=2),
            )
        )
    return traces_returns


def create_volatility_traces(filtered_df: pd.DataFrame, selected_id: str) -> list[go.Scatter]:
    traces_volatility = []
    traces_volatility.append(
        go.Scatter(
            x=filtered_df["DATE"],
            y=filtered_df["VOLATILITY_90D"],
            mode="lines",
            name=f"{selected_id} Volatility 90D",
            marker=dict(color=random_color()),
        )
    )
    return traces_volatility


def create_roc_traces(filtered_df: pd.DataFrame, selected_id: str, selected_days: list[int]) -> list[go.Scatter]:
    traces_roc = []
    traces_roc.append(
        go.Scatter(
            x=filtered_df["DATE"],
            y=filtered_df[f"CLOSE_ROC_{21 if len(selected_days)==0  else selected_days[0]}D"],
            mode="lines",
            name=f"{selected_id} ROC",
            marker=dict(color=random_color()),
        )
    )
    return traces_roc


def create_graph_layout(title: str, xaxis_title: str, yaxis1_title: str, yaxis2_title: str | None = None) -> go.Layout:
    layout = go.Layout(
        title=title,
        xaxis=dict(title=xaxis_title),
        yaxis1=dict(
            title=yaxis1_title,
            side="left",
            showgrid=False,
            type="linear",
            autorange=True,
        ),
    )
    if yaxis2_title:
        layout.yaxis2 = dict(
            title=yaxis2_title,
            side="right",
            overlaying="y",
            type="linear",
            autorange=True,
            showgrid=False,
        )
    return layout


def update_graph(
    selected_ids: list[str], selected_data: list[str], selected_days: list[int], start_date: str, end_date: str, df: pd.DataFrame
) -> tuple[
    dict[str, list[go.Scatter | go.Bar]],
    dict[str, list[go.Histogram | go.Scatter]],
    dict[str, list[go.Scatter]],
    dict[str, list[go.Scatter]],
]:
    traces_main, traces_returns, traces_volatility, traces_roc = [], [], [], []

    for selected_id in selected_ids:
        filtered_df = filter_df(df, selected_id, start_date, end_date)
        traces_main.extend(create_main_traces(filtered_df, selected_id, selected_data, selected_days))
        traces_returns.extend(create_returns_traces(filtered_df, selected_id))
        traces_volatility.extend(create_volatility_traces(filtered_df, selected_id))
        traces_roc.extend(create_roc_traces(filtered_df, selected_id, selected_days))

    return (
        {
            "data": traces_main,
            "layout": create_graph_layout("Stock Data", "Date", "Close Price", "Volume"),
        },
        {
            "data": traces_returns,
            "layout": create_graph_layout("Returns Frequency and KDE", "Returns", "Frequency / Density"),
        },
        {
            "data": traces_volatility,
            "layout": create_graph_layout("Volatility 90D", "Date", "Volatility"),
        },
        {
            "data": traces_roc,
            "layout": create_graph_layout("Rate of Change", "Date", "ROC"),
        },
    )
