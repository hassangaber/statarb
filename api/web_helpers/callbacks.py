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

from api.network.basic_rf import PortfolioPrediction

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
    )
    def update_monte_carlo_simulation(
        n_clicks, selected_stocks, weights, num_days, initial_portfolio
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
                6, num_days, weights, meanReturns, covMatrix, initial_portfolio
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
        portfolio_sims, sharpe_ratios, weight_lists, final_values, var, cvar, spread = (
            MC(mc_sims, T, weights, meanReturns, covMatrix, initial_portfolio)
        )

        if for_plot:

            fig = go.Figure()
            for i in range(mc_sims):
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(T),
                        y=portfolio_sims[:, i],
                        mode="lines",
                        name=f"Simulation {i+1}",
                    )
                )

            best_sim = np.argmax(final_values)
            worst_sim = np.argmin(final_values)
            fig.data[best_sim].line.color = "green"
            fig.data[best_sim].name = "Best Simulation"
            fig.data[worst_sim].line.color = "red"
            fig.data[worst_sim].name = "Worst Simulation"

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
                var,
                cvar,
                spread,
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
        ],
    )
    def update_table(n_clicks, selected_stocks, weights, num_days, initial_portfolio):
        if n_clicks:
            weights = np.array([float(w.strip()) for w in weights.split(",")])
            weights /= np.sum(weights)

            endDate = dt.datetime.now()
            startDate = endDate - dt.timedelta(days=365 * 13)

            (_, meanReturns, covMatrix) = getData(
                df, selected_stocks, start=startDate, end=endDate
            )
            (_, weight_lists, final_values, sharpe_ratios, var, cvar, spread, _) = (
                run_monte_carlo_simulation(
                    2000, num_days, weights, meanReturns, covMatrix, initial_portfolio
                )
            )

            VaR = initial_portfolio - var
            CVaR = initial_portfolio - cvar

            sorted_indices = np.argsort(final_values)[::-1]
            top_bottom_indices = np.concatenate(
                [sorted_indices[:5], sorted_indices[-5:]]
            )

            data = [
                {
                    "Simulation": i + 1,
                    "Final Portfolio Value": f"${final_values[i]:,.2f}",
                    "VaR, CVaR": f"{VaR}, {CVaR}",
                    "Sigma": f"{spread[i]*100:.2f}%",
                    "Sharpe Ratio": f"{sharpe_ratios[i]:.2f}",
                }
                for i in top_bottom_indices
            ]

            columns = [
                {"name": "Simulation", "id": "Simulation"},
                {"name": "Final Portfolio Value", "id": "Final Portfolio Value"},
                {"name": "VaR, CVaR", "id": "VaR, CVaR"},
                {"name": "Sigma", "id": "Sigma"},
                {"name": "Sharpe Ratio", "id": "Sharpe Ratio"},
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
    #State("train-end-date-input", "value"),
    State("test-start-date-input", "value"),
    #State("start-date-input", "value"),
    # State("batch-size-input", "value"),
    # State("epochs-input", "value"),
    # State("learning-rate-input", "value"),
    # State("weight-decay-input", "value"),
    State("initial-investment-input", "value"),
    State("share-volume-input", "value")
    ]
    )
    def handle_model_training(n_clicks, stock_id, 
                              #train_end_date, 
                              test_start_date, 
                              #start_date, 
                              #batch_size, epochs, lr, weight_decay, 
                              initial_investment, share_volume):
        if n_clicks is None or not all([stock_id, 
                                        #train_end_date, 
                                        test_start_date, 
                                        #start_date, 
                                        #batch_size, epochs, lr, weight_decay, 
                                        initial_investment, share_volume]):
            return dash.no_update

        # batch_size = int(batch_size)
        # epochs = int(epochs)
        # lr = float(lr)
        # weight_decay = float(weight_decay)
        initial_investment = int(initial_investment)
        share_volume = int(share_volume)

        model = PortfolioPrediction(
            "assets/data.csv", stock_id, 
            #train_end_date, 
            test_start_date=test_start_date, 
            #start_date,
            #batch_size=batch_size, epochs=epochs, lr=lr, weight_decay=weight_decay,
            initial_investment=initial_investment, share_volume=share_volume
        )
        model.preprocess_data()
        #model.train()
        action_df = model.backtest()

        return action_df.to_json(date_format="iso", orient="split")

    @app.callback(
        [
            Output("portfolio-value-graph", "figure"),
            Output("transaction-signals-graph", "figure"),
            Output("table-container", "children"),
            Output("download-link", "href")
        ],
        Input("stored-data", "data")
    )
    def update_output(data_json):
        if not data_json:
            empty_fig = go.Figure()
            empty_fig.update_layout(title="No data to display", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
            return empty_fig, empty_fig, "Enter parameters and click 'Run Model'"

        action_df = pd.read_json(data_json, orient="split")
        action_df.DATE = pd.to_datetime(action_df.DATE)
        action_df.set_index("DATE", inplace=True)

        weeklyPortfolio = action_df.resample('W').last()
        weeklyPortfolio['RETURNS'] = action_df['RETURNS'].resample('W').mean()
        weeklyPortfolio['alpha'] = action_df['alpha'].resample('W').mean()
        weeklyPortfolio['cumulative_shares'] = action_df['cumulative_shares'].resample('W').sum()
        weeklyPortfolio['cumulative_share_cost'] = action_df['cumulative_share_cost'].resample('W').sum()
        weeklyPortfolio['portfolio_value'] = action_df['portfolio_value'].resample('W').mean()

        weeklyPortfolio['mean_p_buy'] = action_df['p_buy'].resample('W').mean()
        weeklyPortfolio['mean_p_sell'] = action_df['p_sell'].resample('W').mean()
        
        weeklyPortfolio.reset_index(inplace=True)
        action_df.reset_index(inplace=True)

        weeklyPortfolio = weeklyPortfolio[['DATE','CLOSE','RETURNS','mean_p_buy','mean_p_sell','cumulative_shares','cumulative_share_cost','portfolio_value']]
        
        portfolio_value_fig = go.Figure(
            data=[go.Scatter(x=action_df.index, y=action_df['portfolio_value'], mode='lines+markers', name='Portfolio Value')],
            layout=go.Layout(title='Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Portfolio Value')
        )

        transaction_signals_fig = go.Figure(
            data=[
                go.Scatter(x=action_df.index, y=action_df['CLOSE'], mode='lines', name='Close Price'),
                go.Scatter(x=action_df[action_df['predicted_signal'] == 1].index, y=action_df[action_df['predicted_signal'] == 1]['CLOSE'], mode='markers', marker=dict(color='green', size=15), name='Buy Signal', marker_symbol=45),
                go.Scatter(x=action_df[(action_df['predicted_signal'] == 0) & (action_df['cumulative_shares'] > 0)].index, y=action_df[(action_df['predicted_signal'] == 0) & (action_df['cumulative_shares'] > 0)]['CLOSE'], mode='markers', marker=dict(color='red', size=15), name='Sell Signal',marker_symbol=46)
            ],
            layout=go.Layout(title='Transaction Signals Over Time', xaxis_title='Date', yaxis_title='Close Price')
        )

        csv_string = action_df.reset_index().to_csv(index=False, encoding='utf-8')
        csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)

        def format_floats(df):
            float_cols = df.select_dtypes(include=['float']).columns
            for col in float_cols:
                df[col] = df[col].map('{:.4f}'.format)
            return df

        # Apply this formatting function
        formatted_df = format_floats(action_df.copy())

        formatted_df['no_position'] = np.where((formatted_df['predicted_signal'] == 0) & (formatted_df['cumulative_shares'] == 0), 1, 0)
        formatted_df['POSITION'] = np.where((formatted_df['predicted_signal'] == 1), 'BUY','SELL')
        formatted_df['POSITION'] = np.where(formatted_df['no_position']==1, 'NO POSITION', formatted_df['POSITION'])

        formatted_df = formatted_df[['DATE','POSITION','cumulative_shares','portfolio_value','CLOSE','RETURNS','alpha']]

        data_table = dash_table.DataTable(
            data=formatted_df.reset_index().to_dict('records'),
            columns=[{"name": i, "id": i} for i in formatted_df.columns],
            style_table={'overflowX': 'auto'},
            export_format="csv",
            export_headers="display"
        )

        return portfolio_value_fig, transaction_signals_fig, data_table, csv_string