from typing import Any

import dash
import pandas as pd
import plotly.graph_objs as go
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from api.web_helpers.callback.calculate_and_plot_strategy import \
    calculate_and_plot_strategy
from api.web_helpers.callback.handle_mle_model_training import \
    handle_mle_model_training
from api.web_helpers.callback.handle_rf_model_training import \
    handle_rf_model_training
from api.web_helpers.callback.model_training import (handle_model_training,
                                                     update_output)
from api.web_helpers.callback.monte_carlo_simulation import (
    update_monte_carlo_simulation, update_optimal_metrics, update_table)
from api.web_helpers.callback.toggle_ma_selector import toggle_ma_selector
from api.web_helpers.callback.update_graphs import update_graph
from api.web_helpers.callback.update_mle_output import update_mle_output
from api.web_helpers.callback.update_rf_output import update_rf_output


def register_callbacks(app: dash.Dash, df: pd.DataFrame) -> None:
    @app.callback(
        [Output("stock-graph", "figure"),
         Output("returns-graph", "figure"),
         Output("volatility-graph", "figure"),
         Output("roc-graph", "figure")],
        [Input("stock-checklist", "value"),
         Input("data-checklist", "value"),
         Input("date-range-selector", "start_date"),
         Input("date-range-selector", "end_date")]
    )
    def update_graphs(selected_ids, selected_data, start_date, end_date):
        return update_graph(selected_ids, selected_data, start_date, end_date, df)

    @app.callback(
        Output("trades-graph", "figure"),
        [Input("submit-button", "n_clicks")],
        [
            State("stock-input", "value"),
            State("start-date-input", "value"),
            State("end-date-input", "value"),
        ],
    )
    def update_trades_tab(n_clicks: int, stock_id: str, start_date: str, end_date: str) -> go.Figure:
        return calculate_and_plot_strategy(df, stock_id, start_date, end_date)

    @app.callback(
        Output("monte-carlo-simulation-graph", "figure"),
        Input("run-simulation-button", "n_clicks"),
        State("stock-dropdown", "value"),
        State("weights-input", "value"),
        State("num-days-input", "value"),
        State("initial-portfolio-input", "value"),
        State("num-simulations-input", "value"),
    )
    def update_monte_carlo_simulation_callback(
        n_clicks: int, selected_stocks: list[str], weights: str, num_days: int, initial_portfolio: float, num_simulations: int
    ) -> go.Figure:
        return update_monte_carlo_simulation(n_clicks, selected_stocks, weights, num_days, initial_portfolio, num_simulations, df)

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
    def update_table_callback(
        n_clicks: int,
        selected_stocks: list[str],
        weights: str,
        num_days: int,
        initial_portfolio: float,
        num_simulations: int
    ) -> tuple[list[dict], list[dict], go.Figure]:
        return update_table(n_clicks, selected_stocks, weights, num_days, initial_portfolio, num_simulations, df)

    @app.callback(
        Output("optimal-metrics-output", "children"),
        [Input("calculate-metrics-button", "n_clicks")],
        [
            State("target-value-input", "value"),
            State("weights-input", "value"),
            State("initial-portfolio-input", "value"),
            State("num-days-input", "value"),
            State("num-simulations-input", "value"),
            State("stock-dropdown", "value"),
        ],
    )
    def update_optimal_metrics_callback(
        n_clicks: int,
        target_value: float,
        weights: str,
        initial_portfolio: float,
        num_days: int,
        num_simulations:int,
        selected_stocks: list[str]
    ) -> html.Div:
        return update_optimal_metrics(n_clicks, target_value, weights, df, initial_portfolio, num_days, num_simulations, selected_stocks)

    @app.callback(
        Output("stored-data", "data"),
        Input("run-model-button", "n_clicks"),
        [
            State("stock-id-input", "value"),
            State("model-id-input", "value"),
            State("test-start-date-input", "value"),
            State("initial-investment-input", "value"),
            State("share-volume-input", "value"),
        ],
    )
    def handle_model_training_callback(
        n_clicks: int, stock_id: str, model_id: str, test_start_date: str, initial_investment: str, share_volume: str
    ) -> Any:
        return handle_model_training(n_clicks, stock_id, model_id, test_start_date, initial_investment, share_volume)

    @app.callback(
        [
            Output("portfolio-value-graph", "figure"),
            Output("transaction-signals-graph", "figure"),
            Output("cash-on-hand-graph", "figure"),
            Output("pnl-graph", "figure"),
            Output("table-container", "children"),
            Output("download-link", "href"),
            Output("returns-distribution-graph", "figure"),
            Output("stats-container", "children"),
        ],
        Input("stored-data", "data"),
    )
    def update_output_callback(data_json: str) -> tuple[Any, Any, Any, str, Any]:
        return update_output(data_json)

    @app.callback(
        Output("rf-stored-data", "data"),
        Input("rf-run-model-button", "n_clicks"),
        [
            State("rf-stock-id-input", "value"),
            State("rf-model-id-input", "value"),
            State("rf-test-start-date-input", "value"),
            State("rf-initial-investment-input", "value"),
            State("rf-share-volume-input", "value"),
        ],
    )
    def handle_rf_model_training_callback(
        n_clicks: int, stock_id: str, model_id: str, test_start_date: str, initial_investment: str, share_volume: str
    ) -> Any:
        return handle_rf_model_training(n_clicks, stock_id, model_id, test_start_date, initial_investment, share_volume)

    @app.callback(
        [
            Output("rf-portfolio-value-graph", "figure"),
            Output("rf-transaction-signals-graph", "figure"),
            Output("rf-cash-on-hand-graph", "figure"),
            Output("rf-pnl-graph", "figure"),
            Output("rf-table-container", "children"),
            Output("rf-download-link", "href"),
            Output("rf-returns-distribution-graph", "figure"),
            Output("rf-stats-container", "children"),
        ],
        Input("rf-stored-data", "data"),
    )
    def update_rf_output_callback(data_json: str) -> tuple[Any, Any, Any, Any, str, Any, Any, Any]:
        return update_rf_output(data_json)

    @app.callback(
        Output("mle-stored-data", "data"),
        Input("mle-run-model-button", "n_clicks"),
        [
            State("mle-stock-id-input", "value"),
            State("mle-model-id-input", "value"),
            State("mle-test-start-date-input", "value"),
            State("mle-initial-investment-input", "value"),
            State("mle-share-volume-input", "value"),
        ],
    )
    def handle_mle_model_training_callback(
        n_clicks: int, stock_id: str, model_id: str, test_start_date: str, initial_investment: str, share_volume: str
    ) -> Any:
        return handle_mle_model_training(n_clicks, stock_id, model_id, test_start_date, initial_investment, share_volume)

    @app.callback(
        [
            Output("mle-portfolio-value-graph", "figure"),
            Output("mle-transaction-signals-graph", "figure"),
            Output("mle-cash-on-hand-graph", "figure"),
            Output("mle-pnl-graph", "figure"),
            Output("mle-table-container", "children"),
            Output("mle-download-link", "href"),
            Output("mle-returns-distribution-graph", "figure"),
            Output("mle-stats-container", "children"),
        ],
        Input("mle-stored-data", "data"),
    )
    def update_mle_output_callback(data_json: str) -> tuple[Any, Any, Any, Any, str, Any, Any, Any]:
        return update_mle_output(data_json)
