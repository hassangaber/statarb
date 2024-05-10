import numpy as np

def MC(mc_sims:int, 
       T:int, 
       weights:list[float], 
       meanReturns:list[float], 
       covMatrix:np.array, 
       initial_portfolio:int) -> tuple[np.array, list, list, list]:
    
    portfolio_sims = np.zeros((T, mc_sims))
    weight_lists = []
    final_values = []
    sharpe_ratios = []
    risk_free_rate = 0.04 / 252  # daily risk-free rate
    
    for m in range(mc_sims):
        dailyReturns = np.random.multivariate_normal(meanReturns, covMatrix, T)
        portfolio_values = (dailyReturns.dot(weights) + 1).cumprod() * initial_portfolio
        portfolio_sims[:, m] = portfolio_values
        weight_lists.append(weights.tolist())
        final_values.append(portfolio_values[-1])
        std_dev = np.std(dailyReturns.dot(weights))
        mean_return = np.mean(dailyReturns.dot(weights))
        sharpe_ratio = (mean_return - risk_free_rate) / std_dev if std_dev != 0 else 0
        sharpe_ratios.append(sharpe_ratio)
    
    return (portfolio_sims, sharpe_ratios, weight_lists, final_values)