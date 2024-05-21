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
        shares_bought = min(share_volume, P_0["cash_on_hand"] // close_price)

        if shares_bought * close_price <= P_0['cash_on_hand']:
            P["cumulative_shares"] = P_0["cumulative_shares"] + shares_bought
            P["cash_on_hand"] = P_0["cash_on_hand"] - shares_bought * close_price
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
            P["position"] = "buy"
        else:
            P["cumulative_shares"] = P_0["cumulative_shares"]
            P["cash_on_hand"] = P_0["cash_on_hand"]
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
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
            P["cumulative_shares"] = P_0["cumulative_shares"]
            P["cash_on_hand"] = P_0["cash_on_hand"]
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
            P["position"] = "hold"

    return P


def update_portfolio_new(P: pd.Series, P_0: pd.Series, initial_investment: int, share_volume: int, tr:float=0.1) -> pd.Series:
    close_price = P["CLOSE"]
    signal = P["predicted_signal"]

    if (-tr <= signal <= tr):  # Hold signal
        P["cumulative_shares"] = P_0["cumulative_shares"]
        P["cash_on_hand"] = P_0["cash_on_hand"]
        P["share_value"] = P["cumulative_shares"] * close_price
        P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
        P["PnL"] = P["total_portfolio_value"] - initial_investment
        P["position"] = "hold"
    
    elif signal > tr:  # Buy signal
        shares_bought = min(share_volume, P_0["cash_on_hand"] // close_price)

        if shares_bought * close_price <= P_0['cash_on_hand']:
            P["cumulative_shares"] = P_0["cumulative_shares"] + shares_bought
            P["cash_on_hand"] = P_0["cash_on_hand"] - shares_bought * close_price
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
            P["position"] = "buy"
        else:
            P["cumulative_shares"] = P_0["cumulative_shares"]
            P["cash_on_hand"] = P_0["cash_on_hand"]
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
            P["position"] = "hold"
    
    elif signal < -tr:  # Sell signal
        shares_sold = min(share_volume, P_0["cumulative_shares"])
        
        if shares_sold > 0:
            P["cumulative_shares"] = P_0["cumulative_shares"] - shares_sold
            P["cash_on_hand"] = P_0["cash_on_hand"] + shares_sold * close_price
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
            P["position"] = "sell"
        else:
            P["cumulative_shares"] = P_0["cumulative_shares"]
            P["cash_on_hand"] = P_0["cash_on_hand"]
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
            P["position"] = "hold"
    
    return P

def update_portfolio_rf(P: pd.Series, P_0: pd.Series, initial_investment: int, share_volume: int) -> pd.Series:
    close_price = P["CLOSE"]
    signal = P["predicted_signal"]

    if signal == 1:  # Hold signal
        P["cumulative_shares"] = P_0["cumulative_shares"]
        P["cash_on_hand"] = P_0["cash_on_hand"]
        P["share_value"] = P["cumulative_shares"] * close_price
        P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
        P["PnL"] = P["total_portfolio_value"] - initial_investment
        P["position"] = "hold"
    
    elif signal == 2:  # Buy signal
        shares_bought = min(share_volume, P_0["cash_on_hand"] // close_price)

        if shares_bought * close_price <= P_0['cash_on_hand']:
            P["cumulative_shares"] = P_0["cumulative_shares"] + shares_bought
            P["cash_on_hand"] = P_0["cash_on_hand"] - shares_bought * close_price
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
            P["position"] = "buy"
        else:
            P["cumulative_shares"] = P_0["cumulative_shares"]
            P["cash_on_hand"] = P_0["cash_on_hand"]
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
            P["position"] = "hold"
    
    elif signal == 0:  # Sell signal
        shares_sold = min(share_volume, P_0["cumulative_shares"])
        
        if shares_sold > 0:
            P["cumulative_shares"] = P_0["cumulative_shares"] - shares_sold
            P["cash_on_hand"] = P_0["cash_on_hand"] + shares_sold * close_price
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
            P["position"] = "sell"
        else:
            P["cumulative_shares"] = P_0["cumulative_shares"]
            P["cash_on_hand"] = P_0["cash_on_hand"]
            P["share_value"] = P["cumulative_shares"] * close_price
            P["total_portfolio_value"] = P["cash_on_hand"] + P["share_value"]
            P["PnL"] = P["total_portfolio_value"] - initial_investment
            P["position"] = "hold"
    
    return P

