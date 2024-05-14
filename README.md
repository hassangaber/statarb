# Quantitative Investing Dashboard

This repository contains a comprehensive dashboard for analyzing and visualizing the performance of various statistical arbitrage strategies over historical data. The dashboard supports strategies such as momentum, mean reversion, and volatility-based methods.

## Features

- **Data Analysis**: Tools for analyzing stock data, including statistical measures and visualizations.
- **Machine Learning**: Implementations of machine learning models for predictive analysis.
- **Backtesting**: Framework for backtesting different trading strategies.
- **Interactive Dashboard**: User-friendly interface for exploring and understanding the performance of strategies.

## Installation
```bash
git clone https://github.com/hassangaber/statarb.git
cd statarb
pip install -r requirements.txt
gunicorn --preload qstrat:server
```

## Usage

After installation, you can access the dashboard locally. The dashboard allows you to select different stocks, set parameters for backtesting, and visualize the results of various strategies.

Visit the live site at [https://qstrat-e42f91fdc838.herokuapp.com/](https://qstrat-e42f91fdc838.herokuapp.com/)

## Model and Loss Function

### Stock Model

The stock model is a neural network designed to predict stock returns:

$$
\begin{align*}
\text{Layer 1:} & \quad \text{Input} \rightarrow \text{ReLU}(W_1 \cdot \text{Input} + b_1) \\
\text{Layer 2:} & \quad \text{ReLU}(W_2 \cdot \text{Layer 1 Output} + b_2) \\
\text{Output Layer:} & \quad \sigma(W_3 \cdot \text{Layer 2 Output} + b_3)
\end{align*}
$$

where  $\sigma (W^{T} x+b)$  is the sigmoid activation function.

### Volatility Weighted Loss

The custom loss function is defined as:

$$
L(\hat{y}, y) = \text{BCE}(\hat{y}, y) + \lambda \cdot \text{Mean}((\hat{y} - y) \cdot \text{volatility})
$$

where:
-  $\text{BCE}(\hat{y}, y)$  is the binary cross-entropy loss between predictions  $\hat{y}$  and targets  $y$ .
-  $\lambda$  is a weighting factor.
- The second term penalizes large deviations between predictions and targets, scaled by volatility.

## Monte Carlo Simulation

Monte Carlo simulations are used to project the future performance of investment portfolios by running multiple scenarios based on historical data.

### Monte Carlo Simulation

1. Generate random variables  $Z \sim N(0, 1)$
2. Calculate the Cholesky decomposition of the covariance matrix $L$ 
3. Compute daily returns: $\text{dailyReturns} = \mu + L \cdot Z$

4. Compute portfolio values: $\text{portfolioValues} = \text{initialPortfolio} \cdot \prod_{t=1}^{T} (1 + \text{dailyReturns})$

5. Calculate performance metrics (e.g., Sharpe ratio, VaR, CVaR).




