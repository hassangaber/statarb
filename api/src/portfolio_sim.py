import pandas as pd

def update_portfolio(P: pd.Series, P_0: pd.Series, initial_investment: int, share_volume: int) -> pd.Series:
        
        close_price = P["CLOSE"]
        signal = P["predicted_signal"]
        probability = P["p_buy"]
        
        # Hold if probability is between 45% and 55%
        if 0.45 <= probability <= 0.55:
            P["cumulative_shares"] = P_0["cumulative_shares"]
            P["cash_on_hand"] = P_0["cash_on_hand"]
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
            P["position"] = "hold"
            return P
        
        if signal:  # Buy signal
            shares_bought = min(share_volume, P["cash_on_hand"] // close_price)

            if shares_bought * close_price > P['cash_on_hand']:
                shares_bought = 0

            if shares_bought > 0:
                P["cumulative_shares"] = P_0["cumulative_shares"] + shares_bought
                P["cash_on_hand"] = P_0["cash_on_hand"] - shares_bought * close_price
                P["share_value"] = P["cumulative_shares"] * close_price
                P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
                P["PnL"] = P["total_portfolio_value"] - initial_investment
                P["position"] = "buy"
            else:
                P["position"] = "hold"
        
        else:  # Sell signal
            shares_sold = min(share_volume, P_0["cumulative_shares"])
            if shares_sold > 0:
                P["cumulative_shares"] = P_0["cumulative_shares"] - shares_sold
                P["cash_on_hand"] = P_0["cash_on_hand"] + shares_sold * close_price
                P["share_value"] = P["cumulative_shares"] * close_price
                P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
                P["PnL"] = P["total_portfolio_value"] - initial_investment
                P["position"] = "sell"
            else:
                P["position"] = "hold"

        # Update position to hold if no shares bought or sold
        if P["cumulative_shares"] == P_0["cumulative_shares"]:
            P["position"] = "hold"
        
        return P