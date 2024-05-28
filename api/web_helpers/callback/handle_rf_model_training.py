from typing import Any

import dash

from api.network.predict import PortfolioPredictionRF


def handle_rf_model_training(
    n_clicks: int, stock_id: str, model_id: str, test_start_date: str, initial_investment: str, share_volume: str
) -> Any:
    if n_clicks is None or not all([stock_id, test_start_date, model_id, initial_investment, share_volume]):
        return dash.no_update

    initial_investment = int(initial_investment)
    share_volume = int(share_volume)

    model = PortfolioPredictionRF(
        "assets/data.csv",
        stock_id,
        test_start_date=test_start_date,
        initial_investment=initial_investment,
        share_volume=share_volume,
        model_path="assets/xgb.pkl",
    )

    model.preprocess_test_data()
    action_df = model.backtest()

    return action_df.to_json(date_format="iso", orient="split")
