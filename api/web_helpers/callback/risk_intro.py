import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import html

import dash_bootstrap_components as dbc


def calculate_sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def create_risk_profile(data, long_trade_info, short_trade_info):
    # Unpack trade info
    long_prices, long_entry, long_exit, long_entry_price, long_exit_price, long_pl, long_max_loss = long_trade_info
    short_prices, short_entry, short_exit, short_entry_price, short_exit_price, short_pl, short_max_loss = short_trade_info

    # Calculate returns
    long_returns = (long_exit_price - long_entry_price) / long_entry_price
    short_returns = (short_entry_price - short_exit_price) / short_entry_price

    # Calculate Sharpe ratios
    risk_free_rate = 0.041  # 4.1% annual risk-free rate
    daily_risk_free_rate = (1 + risk_free_rate) ** (1/252) - 1
    long_sharpe = calculate_sharpe_ratio(pd.Series(long_returns), daily_risk_free_rate)
    short_sharpe = calculate_sharpe_ratio(pd.Series(short_returns), daily_risk_free_rate)

    # Calculate annualized returns
    days_held_long = long_exit - long_entry
    days_held_short = short_exit - short_entry
    long_annual_return = (1 + long_returns) ** (252 / days_held_long) - 1
    short_annual_return = (1 + short_returns) ** (252 / days_held_short) - 1

    risk_profile = dbc.Row([
        dbc.Col([
            html.H5("Long Trade", className="text-center"),
            html.P(f"Entry Price: ${long_entry_price:.2f}"),
            html.P(f"Exit Price: ${long_exit_price:.2f}"),
            html.P(f"Profit/Loss: ${long_pl:.2f}"),
            html.P(f"Return: {long_returns:.2%}"),
            html.P(f"Annualized Return: {long_annual_return:.2%}"),
            html.P(f"Maximum Drawdown: ${long_max_loss:.2f}"),
            html.P(f"Sharpe Ratio: {long_sharpe:.2f}"),
            html.P(f"Days Held: {days_held_long}")
        ], width=6),
        dbc.Col([
            html.H5("Short Trade", className="text-center"),
            html.P(f"Entry Price: ${short_entry_price:.2f}"),
            html.P(f"Exit Price: ${short_exit_price:.2f}"),
            html.P(f"Profit/Loss: ${short_pl:.2f}"),
            html.P(f"Return: {short_returns:.2%}"),
            html.P(f"Annualized Return: {short_annual_return:.2%}"),
            html.P(f"Maximum Risk: ${short_max_loss:.2f}"),
            html.P(f"Sharpe Ratio: {short_sharpe:.2f}"),
            html.P(f"Days Held: {days_held_short}")
        ], width=6)
    ])

    return risk_profile

def simulate_trade(initial_price, returns, is_long=True):
    trading_days = 252
    simulated_returns = np.random.normal(returns.mean(), returns.std(), trading_days)
    simulated_prices = initial_price * (1 + simulated_returns).cumprod()
    
    entry_day = np.random.randint(0, trading_days // 2)
    exit_day = np.random.randint(entry_day + 1, trading_days)
    
    entry_price = simulated_prices[entry_day]
    exit_price = simulated_prices[exit_day]
    
    if is_long:
        profit_loss = exit_price - entry_price
        max_loss = entry_price - min(simulated_prices[entry_day:exit_day])
    else:
        profit_loss = entry_price - exit_price
        max_loss = max(simulated_prices[entry_day:exit_day]) - entry_price
    
    return simulated_prices, entry_day, exit_day, entry_price, exit_price, profit_loss, max_loss

def create_trade_figure(simulated_prices, entry_day, exit_day, entry_price, exit_price, is_long=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=simulated_prices, mode='lines', name='Simulated Price'))
    fig.add_trace(go.Scatter(x=[entry_day, exit_day], y=[entry_price, exit_price], 
                             mode='markers+lines', name='Trade', line=dict(color='red', width=2)))
    
    fig.add_annotation(x=entry_day, y=entry_price, text="Entry", showarrow=True, arrowhead=1)
    fig.add_annotation(x=exit_day, y=exit_price, text="Exit", showarrow=True, arrowhead=1)
    
    title = "Long Position" if is_long else "Short Position"
    fig.update_layout(title=f"{title}: Entry and Exit", yaxis_title="Price", xaxis_title="Trading Days")
    
    return fig

def risk_intro(data: pd.DataFrame, t: str) -> tuple[html.Div, go.Figure, go.Figure, go.Figure, html.Div]:
    returns = pd.Series(np.diff(np.log(data.CLOSE)))
    
    # Calculate volatility
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days in a year
    
    # Create price and volatility plot
    fig_price_vol = make_subplots(specs=[[{"secondary_y": True}]])
    fig_price_vol.add_trace(go.Scatter(x=returns.index, y=data.CLOSE, name="Price"), secondary_y=False)
    fig_price_vol.add_trace(go.Scatter(x=returns.index, y=data.VOLATILITY_90D * np.sqrt(252), name="Volatility (90-day)"), secondary_y=True)
    fig_price_vol.update_layout(title=f"{t} Price and Volatility", xaxis_title="Date")
    fig_price_vol.update_yaxes(title_text="Price", secondary_y=False)
    fig_price_vol.update_yaxes(title_text="Annualized Volatility", secondary_y=True)
    
    # Simulate long and short positions
    initial_price = data['CLOSE'].iloc[-1]
    
    # Long trade simulation
    long_prices, long_entry, long_exit, long_entry_price, long_exit_price, long_pl, long_max_loss = simulate_trade(initial_price, returns, is_long=True)
    fig_long = create_trade_figure(long_prices, long_entry, long_exit, long_entry_price, long_exit_price, is_long=True)
    
    # Short trade simulation
    short_prices, short_entry, short_exit, short_entry_price, short_exit_price, short_pl, short_max_loss = simulate_trade(initial_price, returns, is_long=False)
    fig_short = create_trade_figure(short_prices, short_entry, short_exit, short_entry_price, short_exit_price, is_long=False)
    
    # Calculate risk metrics
    long_risk_reward = long_pl / long_max_loss if long_max_loss != 0 else np.inf
    short_risk_reward = short_pl / short_max_loss if short_max_loss != 0 else np.inf
    
    volatility_output = html.Div([
        html.P(f"Daily Volatility: {daily_volatility:.2%}"),
        html.P(f"Annualized Volatility: {annualized_volatility:.2%}")
    ])
    

    # In risk_intro function
    long_trade_info = (long_prices, long_entry, long_exit, long_entry_price, long_exit_price, long_pl, long_max_loss)
    short_trade_info = (short_prices, short_entry, short_exit, short_entry_price, short_exit_price, short_pl, short_max_loss)
    risk_profile = create_risk_profile(data, long_trade_info, short_trade_info)

    
    return volatility_output, fig_price_vol, fig_long, fig_short, risk_profile
