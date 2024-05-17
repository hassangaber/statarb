import numpy as np
import pandas as pd
import datetime as dt
import urllib

import dash
from dash import callback_context, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from scipy.stats import gaussian_kde

from api.network.predict import PortfolioPrediction

from api.src.compute import compute_signal, getData
from api.src.monteCarlo import MC
from api.web_helpers.utils import random_color, kde_scipy

def register_callbacks(app: dash.Dash, df: pd.DataFrame) -> None:
    """TIME SERIES ANALYSIS CALLBACK"""

    @app.callback(Output("ma-selector", "style"), [Input("data-checklist", "value")])
    def toggle_ma_selector(selected_data):
        if "SMA" in selected_data or "EWMA" in selected_data:
            return {"display": "block"}
        else:
            return {"display": "none"}

    @app.callback(
        [
            Output("stock-graph", "figure"),
            Output("returns-graph", "figure"),
            Output("volatility-graph", "figure"),
            Output("roc-graph", "figure"),
        ],
        [
            Input("stock-checklist", "value"),
            Input("data-checklist", "value"),
            Input("ma-day-dropdown", "value"),
            Input("date-range-selector", "start_date"),
            Input("date-range-selector", "end_date"),
        ],
    )
    def update_graph(
        selected_ids: list[str],
        selected_data: list[str],
        selected_days: list[int],
        start_date: str,
        end_date: str,
    ) -> tuple[dict, dict, dict, dict]:

        traces_main, traces_returns, traces_volatility, traces_roc = [], [], [], []

        for selected_id in selected_ids:

            filtered_df = df[df["ID"] == selected_id]
            filtered_df = filtered_df[
                (filtered_df["DATE"] >= start_date) & (filtered_df["DATE"] <= end_date)
            ]

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

            if "RETURNS" in selected_data:

                total_counts = len(filtered_df["RETURNS"].dropna())
                bin_size = (
                    np.max(filtered_df["RETURNS"]) - np.min(filtered_df["RETURNS"])
                ) / 40
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

                x_grid = np.linspace(
                    np.min(filtered_df["RETURNS"]), np.max(filtered_df["RETURNS"]), 1000
                )
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

            traces_volatility.append(
                go.Scatter(
                    x=filtered_df["DATE"],
                    y=filtered_df["VOLATILITY_90D"],
                    mode="lines",
                    name=f"{selected_id} Volatility 90D",
                    marker=dict(color=random_color()),
                )
            )

            traces_roc.append(
                go.Scatter(
                    x=filtered_df["DATE"],
                    y=filtered_df[
                        f"CLOSE_ROC_{21 if len(selected_days)==0  else selected_days[0]}D"
                    ],
                    mode="lines",
                    name=f"{selected_id} ROC",
                    marker=dict(color=random_color()),
                )
            )

        return (
            {
                "data": traces_main,
                "layout": go.Layout(
                    title="Stock Data",
                    xaxis=dict(title="Date"),
                    yaxis1=dict(
                        title="Close Price",
                        side="left",  # Close price on the left y-axis
                        showgrid=False,
                        type="linear",  # Linear scale
                        autorange=True,  # Automatically adjust the range based on the data
                    ),
                    yaxis2=dict(
                        title="Volume",
                        side="right",  # Volume on the right y-axis
                        overlaying="y",  # This specifies that yaxis2 is overlaying yaxis
                        type="linear",  # Linear scale
                        autorange=True,  # Automatically adjust the range based on the data
                        showgrid=False,
                    ),
                ),
            },
            {
                "data": traces_returns,
                "layout": go.Layout(
                    title="Returns Frequency and KDE",
                    xaxis=dict(title="Returns"),
                    yaxis=dict(title="Frequency / Density"),
                ),
            },
            {
                "data": traces_volatility,
                "layout": go.Layout(
                    title="Volatility 90D",
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Volatility"),
                ),
            },
            {
                "data": traces_roc,
                "layout": go.Layout(
                    title="Rate of Change",
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="ROC"),
                ),
            },
        )

    """BACKTEST WITH INDICATORS CALLBACK"""

    @app.callback(
        Output("trades-graph", "figure"),
        [Input("submit-button", "n_clicks")],
        [
            State("stock-input", "value"),
            State("start-date-input", "value"),
            State("end-date-input", "value"),
        ],
    )
    def update_trades_tab(n_clicks, stock_id, start_date, end_date):
        fig = calculate_and_plot_strategy(df, stock_id, start_date, end_date)
        return fig

    def calculate_and_plot_strategy(
        df: pd.DataFrame, stock_id: str, start_date: str, end_date: str
    ):
        """Calculate trading signals and plot results using Plotly."""
        M = df.loc[df.ID == stock_id]
        M = M[
            (M.DATE >= pd.to_datetime(start_date))
            & (M.DATE <= pd.to_datetime(end_date))
        ]
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

        # Add price, SMA lines, and entry signals to the first plot
        fig.add_trace(
            go.Scatter(x=M["DATE"], y=M["CLOSE"], mode="lines", name="Close Price"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=M["DATE"], y=M["CLOSE_SMA_9D"], mode="lines", name="9-day SMA"
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=M["DATE"], y=M["CLOSE_SMA_21D"], mode="lines", name="21-day SMA"
            ),
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

        # Plot cumulative returns for the strategy and buy-and-hold
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

    """MONTE-CARLO CALLBACK"""

    @app.callback(
    Output("monte-carlo-simulation-graph", "figure"),
    Input("run-simulation-button", "n_clicks"),
    State("stock-dropdown", "value"),
    State("weights-input", "value"),
    State("num-days-input", "value"),
    State("initial-portfolio-input", "value"),
    State("num-simulations-input", "value"),
    )
    def update_monte_carlo_simulation(
        n_clicks, selected_stocks, weights, num_days, initial_portfolio, num_simulations
    ):
        triggered = callback_context.triggered[0]

        if triggered["value"] and n_clicks > 0:
            weights = np.array([float(w.strip()) for w in weights.split(",")])
            weights /= np.sum(weights)

            endDate = dt.datetime.now()
            startDate = endDate - dt.timedelta(days=365 * 10)

            _, meanReturns, covMatrix = getData(
                df, selected_stocks, start=startDate, end=endDate
            )

            res = run_monte_carlo_simulation(
                num_simulations, num_days, weights, meanReturns, covMatrix, initial_portfolio
            )

            return res[0]
        return go.Figure()

    def run_monte_carlo_simulation(
        mc_sims,
        T,
        weights,
        meanReturns,
        covMatrix,
        initial_portfolio,
        for_plot: bool = True,
    ) -> tuple:
        global portfolio_sims
        portfolio_sims, sharpe_ratios, weight_lists, final_values, VaR_list, CVaR_list, sigmas, sortino_ratios = (
            MC(mc_sims, T, weights, meanReturns, covMatrix, initial_portfolio)
        )

        if for_plot:
            fig = go.Figure()

            sorted_indices = np.argsort(final_values)[::-1]  # Sort indices in descending order
            top_bottom_indices = np.concatenate([sorted_indices[:5], sorted_indices[-5:]])  # Top 5 and bottom 5 indices

            for i in top_bottom_indices:
                color = "green" if i in sorted_indices[:5] else "red"
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(T),
                        y=portfolio_sims[:, i],
                        mode="lines",
                        name=f"Simulation {i+1}",
                        line=dict(color=color)
                    )
                )

            # best_sim = np.argmax(final_values)
            # worst_sim = np.argmin(final_values)
            # fig.data[best_sim].line.color = "green"
            # fig.data[best_sim].name = "Best Simulation"
            # fig.data[worst_sim].line.color = "red"
            # fig.data[worst_sim].name = "Worst Simulation"

            fig.update_layout(
                title="Monte Carlo Portfolio Simulation Over Time",
                xaxis_title="Days",
                yaxis_title="Portfolio Value",
                legend_title="Simulation",
            )

            return (
                fig,
                weight_lists,
                final_values,
                sharpe_ratios,
                VaR_list,
                CVaR_list,
                sigmas,
                sortino_ratios,
                portfolio_sims,
            )
        else:
            return portfolio_sims

    @app.callback(
        [
            Output("simulation-results-table", "data"),
            Output("simulation-results-table", "columns"),
            Output("distribution-graph", "figure"),
        ],
        [Input("run-simulation-button", "n_clicks")],
        [
            State("stock-dropdown", "value"),
            State("weights-input", "value"),
            State("num-days-input", "value"),
            State("initial-portfolio-input", "value"),
            State("num-simulations-input", "value"),
        ],
    )
    def update_table(n_clicks, selected_stocks, weights, num_days, initial_portfolio, num_simulations):
        if n_clicks:
            weights = np.array([float(w.strip()) for w in weights.split(",")])
            weights /= np.sum(weights)

            endDate = dt.datetime.now()
            startDate = endDate - dt.timedelta(days=365 * 13)

            (_, meanReturns, covMatrix) = getData(
                df, selected_stocks, start=startDate, end=endDate
            )
            (_, weight_lists, final_values, sharpe_ratios, VaR_list, CVaR_list, sigmas, sortino_ratios, _) = (
                run_monte_carlo_simulation(
                    num_simulations, num_days, weights, meanReturns, covMatrix, initial_portfolio
                )
            )

            sorted_indices = np.argsort(final_values)[::-1]
            top_bottom_indices = np.concatenate(
                [sorted_indices[:5], sorted_indices[-5:]]
            )

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

            fig = ff.create_distplot(
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

    @app.callback(
        Output("optimal-metrics-output", "children"),
        [Input("calculate-metrics-button", "n_clicks")],
        [
            State("target-value-input", "value"),
            State("monte-carlo-simulation-graph", "figure"),
            State("weights-input", "value"),
            State("initial-portfolio-input", "value"),
        ],
    )
    def update_optimal_metrics(
        n_clicks, target_value, simulation_figure, weights, initial_portfolio
    ):
        if n_clicks is not None:
            weights = np.array([float(w.strip()) for w in weights.split(",")])
            weights /= np.sum(weights)
            breakout_above = target_value
            breakout_below = initial_portfolio * 0.8

            holding_periods = []
            upper_breakouts = []
            lower_breakouts = []

            for sim in portfolio_sims.T:
                valid_periods = np.where(sim >= breakout_above)[0]
                if valid_periods.size > 0:
                    holding_periods.append(valid_periods[0])  # Get the first index where the condition is met
                upper_breakouts.append(np.max(sim))
                lower_breakouts.append(np.min(sim))

            optimal_holding_period = np.max(holding_periods) if holding_periods else "Target not reached in any simulation"
            upper_prob = np.sum(np.array(upper_breakouts) > breakout_above) / len(portfolio_sims.T)
            lower_prob = np.sum(np.array(lower_breakouts) < breakout_below) / len(portfolio_sims.T)

            return html.Div(
                [
                    html.P(f"Optimal Holding Period: {optimal_holding_period} days"),
                    html.P(
                        f"Breakout Probabilities: {upper_prob * 100:.2f}% above ${breakout_above}, {lower_prob * 100:.2f}% below ${breakout_below}"
                    ),
                ]
            )
        return "Please run the simulation and set a target value to calculate metrics."


    @app.callback(
    Output("stored-data", "data"),
    Input("run-model-button", "n_clicks"),
    [
    State("stock-id-input", "value"),
    State("model-id-input", "value"),
    State("test-start-date-input", "value"),
    State("initial-investment-input", "value"),
    State("share-volume-input", "value")
    ]
    )
    def handle_model_training(n_clicks, 
                              stock_id, 
                              model_id,
                              test_start_date,
                              initial_investment, 
                              share_volume):
        
        if n_clicks is None or not all([stock_id,
                                        test_start_date,
                                        model_id,
                                        initial_investment, 
                                        share_volume]):
            return dash.no_update

        initial_investment = int(initial_investment)
        share_volume = int(share_volume)

        model = PortfolioPrediction(
            "assets/data.csv", 
            stock_id, 
            test_start_date=test_start_date,
            initial_investment=initial_investment, 
            share_volume=share_volume
        )

        model.preprocess_test_data()
        action_df = model.backtest(model_id=model_id)

        return action_df.to_json(date_format="iso", orient="split")

    @app.callback(
        [
            Output("portfolio-value-graph", "figure"),
            Output("transaction-signals-graph", "figure"),
            Output("cash-on-hand-graph", "figure"),
            Output("pnl-graph", "figure"),
            Output("table-container", "children"),
            Output("download-link", "href"),
            Output("returns-distribution-graph", "figure"),
            Output("stats-container", "children")
        ],
            Input("stored-data", "data")
        )
    def update_output(data_json):
        if not data_json:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data to display", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            return empty_fig, empty_fig, "Enter parameters and click 'Run Model'", "", empty_fig

        action_df = pd.read_json(data_json, orient="split")
        action_df.DATE = pd.to_datetime(action_df.DATE)

        # Define the layout for smoother and more appealing graphs
        graph_layout = go.Layout(
            title='',
            xaxis=dict(title='Date', showgrid=True, zeroline=False),
            yaxis=dict(title='', showgrid=True, zeroline=False),
            hovermode='closest',
            template='plotly_white'
        )

        # Portfolio Value Graph
        portfolio_value_fig = go.Figure(
            data=[
                go.Scatter(x=action_df.DATE, y=action_df['total_portfolio_value'], mode='lines', name='Portfolio Value', line=dict(shape='spline', smoothing=1.3)),
                #go.Scatter(x=action_df.DATE, y=action_df['cash_on_hand'], mode='lines', name='Cash on Hand', line=dict(shape='spline', smoothing=1.3)),
                #go.Scatter(x=action_df.DATE, y=action_df['PnL'], mode='lines', name='PnL', line=dict(shape='spline', smoothing=1.3))
            ],
            layout=graph_layout.update(title='Portfolio Value Over Time', yaxis_title='Value ($)')
        )

        cash_on_hand_fig = go.Figure(
            data=[
                go.Scatter(x=action_df.DATE, y=action_df['cash_on_hand'], mode='lines', name='Cash on Hand', line=dict(shape='spline', smoothing=1.3))
            ],
            layout=graph_layout.update(title='Cash on Hand Over Time', yaxis_title='Value ($)')
        )

        
        pnl_fig = go.Figure(
            data=[
                go.Scatter(x=action_df.DATE, y=action_df['PnL'], mode='lines', name='Profit and Loss', line=dict(shape='linear', smoothing=1.3)),
                go.Scatter(x=action_df.DATE, y=[0]*len(action_df), mode='lines', name='Break-Even Level', line=dict(color='red', dash='dash'))
            ],
            layout=graph_layout.update(title='PnL Over Time', yaxis_title='Value ($)')
        )

        # Transaction Signals Graph
        transaction_signals_fig = go.Figure(
            data=[
                go.Scatter(x=action_df.DATE, y=action_df['CLOSE'], mode='lines', name='Close Price', line=dict(shape='linear', smoothing=1.3)),
                go.Scatter(x=action_df[action_df['predicted_signal'] >0.1].DATE, y=action_df[action_df['predicted_signal'] >0.1]['CLOSE'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'),
                go.Scatter(x=action_df[(action_df['predicted_signal'] < -0.1) & (action_df['cumulative_shares'] > 0)].DATE, y=action_df[(action_df['predicted_signal']< -0.1) & (action_df['cumulative_shares'] > 0)]['CLOSE'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal')
            ],
            layout=graph_layout.update(title='Transaction Signals Over Time', yaxis_title='Close Price (US$)')
        )

        # CSV Download Link
        csv_string = action_df.reset_index().to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

        # Format Floats
        def format_floats(df):
            float_cols = df.select_dtypes(include=['float']).columns
            for col in float_cols:
                df[col] = df[col].map('{:.4f}'.format)
            return df

        formatted_df = format_floats(action_df.copy())

        # Data Table with compact formatting and bold column names
        data_table = dash_table.DataTable(
            data=formatted_df.reset_index().to_dict('records'),
            columns=[{"name": i, "id": i, "deletable": False, "renamable": False, "editable": False} for i in formatted_df.columns],
            style_table={'overflowX': 'auto', 'maxHeight': '500px', 'overflowY': 'auto'},
            style_cell={'padding': '5px', 'whiteSpace': 'normal', 'height': 'auto', 'fontSize': 12},
            style_header={'fontWeight': 'bold', 'fontSize': 14},
            export_format="csv",
            export_headers="display"
        )

        # Distribution of Returns
        returns = action_df['RETURNS'].dropna()
        kde = gaussian_kde(returns)
        x = np.linspace(returns.min(), returns.max(), 1000)
        y = kde(x)

        returns_distribution_fig = go.Figure(
            data=[
                go.Histogram(x=returns, nbinsx=150, name='Returns Histogram', opacity=0.5, histnorm='probability density'),
                go.Scatter(x=x, y=y, mode='lines', name='Returns KDE', line=dict(shape='spline', smoothing=1.3))
            ],
            layout=graph_layout.update(title='KDE and Histogram of Returns', xaxis_title='Returns', yaxis_title='Density')
        )
        returns_distribution_fig.update_layout(height=300)

        # Mean and Standard Deviation of Returns
        mean_return = returns.mean()
        std_return = returns.std()

        stats_container = html.Div(
            [
                html.P(f"Mean Return: {mean_return:.4f}", style={'font-size': '18px'}),
                html.P(f"Standard Deviation of Returns: {std_return:.4f}", style={'font-size': '18px'})
            ]
        )

        return portfolio_value_fig, transaction_signals_fig, cash_on_hand_fig, pnl_fig, data_table, csv_string, returns_distribution_fig, stats_container

