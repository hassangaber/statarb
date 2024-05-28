import numpy as np
import pandas as pd

def mcVaR(returns: pd.Series, alpha: int = 5) -> float:
    """Calculate the Value at Risk (VaR) at the given alpha percentile."""
    var_value = np.percentile(returns, alpha)
    return var_value

def mcCVaR(returns: pd.Series, alpha: int = 5) -> float:
    """Calculate the Conditional Value at Risk (CVaR) at the given alpha percentile."""
    var_value = mcVaR(returns, alpha)
    belowVaR = returns <= var_value
    cvar_value = returns[belowVaR].mean()
    return cvar_value

def MC(
    mc_sims: int,
    T: int,
    weights: np.ndarray,
    meanReturns: np.ndarray,
    covMatrix: np.ndarray,
    initial_portfolio: float,
) -> tuple[np.ndarray, list, list, list, list, list, list, list, list, list]:
    """
    Perform Monte Carlo simulation for portfolio returns.

    Parameters:
        mc_sims (int): Number of Monte Carlo simulations.
        T (int): Time horizon in days.
        weights (np.ndarray): Portfolio weights.
        meanReturns (np.ndarray): Mean returns for the assets.
        covMatrix (np.ndarray): Covariance matrix of asset returns.
        initial_portfolio (float): Initial portfolio value.

    Returns:
        tuple: Simulated portfolio values, Sharpe ratios, weight lists, final values, VaR, CVaR, sigmas, Sortino ratios, drawdowns, calmar ratios.
    """
    portfolio_sims = np.zeros((T, mc_sims))
    meanM = np.full((T, len(weights)), meanReturns).T

    weight_lists = []
    final_values = []
    sharpe_ratios = []
    sortino_ratios = []
    sigmas = []
    VaR_list = []
    CVaR_list = []
    risk_free_rate = 0.041 / 252  # daily risk-free rate

    L = np.linalg.cholesky(covMatrix)

    for m in range(mc_sims):
        Z = np.random.normal(size=(T, len(weights)))
        dailyReturns = meanM + np.inner(L, Z)
        portfolio_values = np.cumprod(np.dot(weights, dailyReturns) + 1) * initial_portfolio
        portfolio_sims[:, m] = portfolio_values

        std_dev = np.std(np.dot(dailyReturns.T, weights))
        sigmas.append(std_dev)
        mean_return = np.mean(np.dot(dailyReturns.T, weights))
        sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0

        downside_returns = np.dot(dailyReturns.T, weights)[np.dot(dailyReturns.T, weights) < 0]
        sortino_ratio = (mean_return - risk_free_rate) / np.std(downside_returns) if np.std(downside_returns) != 0 else 0

        sharpe_ratios.append(sharpe_ratio)
        sortino_ratios.append(sortino_ratio)
        weight_lists.append(weights.tolist())
        final_values.append(portfolio_values[-1])

        returns_series = pd.Series(portfolio_values)
        VaR_list.append(mcVaR(returns_series))
        CVaR_list.append(mcCVaR(returns_series))

    return (
        portfolio_sims,
        sharpe_ratios,
        weight_lists,
        final_values,
        VaR_list,
        CVaR_list,
        sigmas,
        sortino_ratios,
    )
