import numpy as np
import pandas as pd

def mcVaR(returns:pd.Series, alpha:int=5) -> list[float]:
    return np.percentile(returns, alpha)

def mcCVaR(returns:pd.Series, alpha:int=5) -> list[float]:
    belowVaR = returns <= mcVaR(returns, alpha=alpha)
    return returns[belowVaR].mean()

def MC(mc_sims:int, 
       T:int, 
       weights:list[float], 
       meanReturns:list[float], 
       covMatrix:np.array, 
       initial_portfolio:int
    ) -> tuple[np.array, list, list, list, float, float, list]:
    
    portfolio_sims = np.zeros((T, mc_sims))
    meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
    meanM = meanM.T

    weight_lists = []
    final_values = []
    sharpe_ratios = []
    sigmas = []
    risk_free_rate = 0.041 / 252  # daily risk-free rate
    
    for m in range(mc_sims):

        Z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(covMatrix)

        dailyReturns = meanM + np.inner(L, Z)
        portfolio_values = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initial_portfolio
        portfolio_sims[:,m] = portfolio_values
    
        std_dev = np.std(np.dot(dailyReturns.T,weights))
        sigmas.append(std_dev)
        mean_return = np.mean(dailyReturns.T.dot(weights))
        sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0
        
        sharpe_ratios.append(sharpe_ratio)
        weight_lists.append(weights.tolist())
        final_values.append(portfolio_values[-1])

    final_values_series = pd.Series(portfolio_sims[-1,:])
    var = mcVaR(final_values_series)
    cvar = mcCVaR(final_values_series)
    
    return (portfolio_sims, sharpe_ratios, weight_lists, final_values, var, cvar, sigmas)
    