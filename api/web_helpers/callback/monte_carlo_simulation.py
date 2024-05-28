import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import html
from plotly.figure_factory import create_distplot

from api.src.compute import getData
from api.src.monteCarlo import MC


# Monte Carlo Simulation Function
def update_monte_carlo_simulation(
    n_clicks: int, selected_stocks: list[str], weights: str, num_days: int, initial_portfolio: float, num_simulations: int, df: pd.DataFrame
) -> go.Figure:
    if n_clicks:
        weights = np.array([float(w.strip()) for w in weights.split(",")])
        weights /= np.sum(weights)
        num_days = int(num_days)
        num_simulations = int(num_simulations)
        initial_portfolio = float(initial_portfolio)

        endDate = dt.datetime.now()
        startDate = endDate - dt.timedelta(days=365 * 10)

        _, meanReturns, covMatrix = getData(df, selected_stocks, start=startDate, end=endDate)

        portfolio_sims, sharpe_ratios, weight_lists, final_values, VaR_list, CVaR_list, sigmas, sortino_ratios = MC(
            num_simulations, num_days, weights, meanReturns, covMatrix, initial_portfolio
        )

        fig = go.Figure()

        sorted_indices = np.argsort(final_values)[::-1] 
        top_bottom_indices = np.concatenate([sorted_indices[:10], sorted_indices[-10:]]) 

        for i in top_bottom_indices:
                color = "blue" if i in sorted_indices[:10] else "red"
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(num_days), y=portfolio_sims[:, i], mode="lines", name=f"Simulation {i+1}", line=dict(color=color), opacity=0.4
                    )
                )

        mean_simulation = portfolio_sims.mean(axis=1)
        fig.add_trace(
            go.Scatter(x=np.arange(num_days), y=mean_simulation, mode="lines", name="Mean", line=dict(color='black', width=1))
        )

        fig.update_layout(
            title="Monte Carlo Portfolio Simulation Over Time",
            xaxis_title="Days",
            yaxis_title="Portfolio Value",
            legend_title="Simulation",
            template='plotly_white'
        )

        return fig
    return go.Figure()

# Update Optimal Metrics Function
def update_optimal_metrics(
    n_clicks: int, target_value: float, weights: str,df:pd.DataFrame, initial_portfolio: float, num_days: int, num_simulations:int, selected_stocks: list[str]
) -> html.Div:
    if n_clicks:
        weights = np.array([float(w.strip()) for w in weights.split(",")])
        weights /= np.sum(weights)
        breakout_above = target_value
        breakout_below = initial_portfolio * 0.8

        holding_periods = []
        upper_breakouts = []
        lower_breakouts = []

        endDate = dt.datetime.now()
        startDate = endDate - dt.timedelta(days=365 * 13)

        _, meanReturns, covMatrix = getData(df, selected_stocks, start=startDate, end=endDate)

        portfolio_sims, sharpe_ratios, weight_lists, final_values, VaR_list, CVaR_list, sigmas, sortino_ratios= MC(
            num_simulations, num_days, weights, meanReturns, covMatrix, initial_portfolio
        )

        for sim in portfolio_sims.T:
            valid_periods = np.where(sim >= breakout_above)[0]
            if valid_periods.size > 0:
                holding_periods.append(valid_periods[0])
            upper_breakouts.append(np.max(sim))
            lower_breakouts.append(np.min(sim))

        optimal_holding_period = np.max(holding_periods) if holding_periods else "Target not reached in any simulation"
        upper_prob = np.sum(np.array(upper_breakouts) > breakout_above) / len(portfolio_sims.T)
        lower_prob = np.sum(np.array(lower_breakouts) < breakout_below) / len(portfolio_sims.T)
        lower_prob_ = np.sum(np.array(lower_breakouts) < (initial_portfolio*0.5)) / len(portfolio_sims.T)

        return html.Div(
            [
                html.P(f"Optimal Holding Period: {optimal_holding_period} days"),
                html.P(
                    f"Breakout Probabilities: {upper_prob * 100:.2f}% above ${breakout_above}, \
                    {lower_prob * 100:.2f}% below ${breakout_below} \
                    {lower_prob_ * 100:.2f}% below ${initial_portfolio*0.5}"
                ),
            ]
        )
    else:
        return html.Div("Please run the simulation and set a target value to calculate metrics.")

# Update Table Function
def update_table(
    n_clicks: int,
    selected_stocks: list[str],
    weights: str,
    num_days: int,
    initial_portfolio: float,
    num_simulations: int,
    df: pd.DataFrame,
) -> tuple[list[dict], list[dict], go.Figure]:
    if n_clicks:
        weights = np.array([float(w.strip()) for w in weights.split(",")])
        weights /= np.sum(weights)
        num_days = int(num_days)
        num_simulations = int(num_simulations)
        initial_portfolio = float(initial_portfolio)

        endDate = dt.datetime.now()
        startDate = endDate - dt.timedelta(days=365 * 13)

        _, meanReturns, covMatrix = getData(df, selected_stocks, start=startDate, end=endDate)

        portfolio_sims, sharpe_ratios, weight_lists, final_values, VaR_list, CVaR_list, sigmas, sortino_ratios= MC(
            num_simulations, num_days, weights, meanReturns, covMatrix, initial_portfolio
        )

        sorted_indices = np.argsort(final_values)[::-1] 
        top_bottom_indices = np.concatenate([sorted_indices[:10], sorted_indices[-10:]]) 

        data = [
            {
                "Simulation": i + 1,
                "Final Portfolio Value": f"${final_values[i]:,.2f}",
                "VaR": f"${VaR_list[i]:,.2f}",
                "CVaR": f"${CVaR_list[i]:,.2f}",
                "Sigma": f"{sigmas[i]*100:.2f}%",
                "Sharpe Ratio": f"{sharpe_ratios[i]:.2f}",
                "Sortino Ratio": f"{sortino_ratios[i]:.2f}",
            }
            for i in top_bottom_indices
        ]

        columns = [
            {"name": "Simulation", "id": "Simulation"},
            {"name": "Final Portfolio Value", "id": "Final Portfolio Value"},
            {"name": "VaR", "id": "VaR"},
            {"name": "CVaR", "id": "CVaR"},
            {"name": "Sigma", "id": "Sigma"},
            {"name": "Sharpe Ratio", "id": "Sharpe Ratio"},
            {"name": "Sortino Ratio", "id": "Sortino Ratio"},
        ]

        fig = create_distplot(
            [final_values],
            ["Portfolio Values"],
            bin_size=[(np.max(final_values) - np.min(final_values)) / 200],
            show_hist=True,
            show_rug=False,
            curve_type="kde",
        )

        mean_value = np.mean(final_values)
        std_dev = np.std(final_values)
        median_value = np.median(final_values)

        fig.update_layout(
            title="Distribution of Final Portfolio Values",
            xaxis_title="Final Portfolio Value",
            yaxis_title="Density",
            plot_bgcolor="white",
            template='plotly_white',
            annotations=[
                dict(
                    x=mean_value,
                    y=0,
                    xref="x",
                    yref="paper",
                    text="Mean: {:.2f}".format(mean_value),
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-90,
                    bordercolor="black",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="white",
                    opacity=0.8,
                ),
                dict(
                    x=median_value,
                    y=0,
                    xref="x",
                    yref="paper",
                    text="Median: {:.2f}".format(median_value),
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-60,
                    bordercolor="black",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="white",
                    opacity=0.8,
                ),
                dict(
                    x=mean_value + std_dev,
                    y=0,
                    xref="x",
                    yref="paper",
                    text="+1 Z Score: {:.2f}".format(mean_value + std_dev),
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-80,
                    bordercolor="black",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="white",
                    opacity=0.8,
                ),
                dict(
                    x=mean_value - std_dev,
                    y=0,
                    xref="x",
                    yref="paper",
                    text="-1 Z Score: {:.2f}".format(mean_value - std_dev),
                    showarrow=True,
                    arrowhead=1,
                    ax=0,
                    ay=-100,
                    bordercolor="black",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="white",
                    opacity=0.8,
                ),
            ],
        )

        return (data, columns, fig)

    return ([], [], go.Figure())



