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

## Constructing the Target for Predicting Changes in Returns

In supervised learning, labeling is necessary to train models to predict future changes in returns. This dataset class creates a target label for predicting changes in returns based on a dynamic threshold calculated from rolling volatility. The labels are classified into three categories:
- **1**: Change in returns greater than the positive threshold.
- **-1**: Change in returns less than the negative threshold.
- **0**: Change in returns within the threshold range.

## Labeling Method

### Change in Returns Calculation
The change in returns ($$\Delta r_t$$) over a specified horizon ($$h$$) is calculated as:

$$ 
\Delta r_t = r_t - r_{t-h} 
$$

where $$r_t$$ is the return at time $$t$$. We don't calculate future returns as that would introduce look-ahead bias to the data.

### Rolling Indicators
In addition to returns, several rolling indicators are calculated to enhance the predictive power of the model:

- **Momentum**: Calculated as the mean of returns over the horizon:

$$
M_t = \frac{1}{h} \sum_{i=t-h+1}^{t} r_i
$$

- **Simple Moving Averages (SMA)**: For different periods to capture trends:

$$
S_{9} = \frac{1}{9} \sum_{i=t-9+1}^{t} \cdot P_i
$$

$$
S_{21} = \frac{1}{21} \sum_{i=t-21+1}^{t} \cdot P_i
$$

### Dynamic Threshold
The threshold ($$T$$) is defined as a multiple of the rolling volatility:

$$
B_t = T \cdot V_t 
$$

where $$V_t$$ is the rolling volatility over a 90-day period.

### Label Assignment
Labels are assigned based on the future returns and the calculated indicators:

- **Buy Signal (1)**: Assigned when:
    - Future returns are greater than the positive threshold.
    - Momentum is positive.
    - 9-day SMA is greater than 21-day SMA.

- **Sell Signal (-1)**: Assigned when:
    - Future returns are less than the negative threshold.
    - Momentum is negative.
    - 9-day SMA is less than 21-day SMA.

- **Hold Signal (0)**: Assigned when the conditions for buy and sell signals are not met.

The implementation in the `TimeSeriesDataset` class involves the following steps:

1. **Initialize the Class**: Set the parameters and prepare the data.
2. **Preprocess the Data**: Calculate changes in returns, rolling indicators, and assign labels.
3. **Scale Features**: Normalize the features using `StandardScaler`.
4. **Data Handling**: Implement methods to get the length of the dataset and retrieve individual data points.

## Model Architecture and Loss Function

### Convolutional Neural Network (CNN)
The trading signal model is based on a convolutional neural network (CNN) which captures temporal patterns in the data. The architecture includes:
- **Conv1D Layers**: To capture temporal dependencies in the time series data.
- **Adaptive Pooling**: To reduce the sequence length to a fixed size.
- **Fully Connected Layers**: To further process the extracted features.
- **Activation Functions**: `ReLU` between layers and `Tanh` at the output to constrain the signals between -1 and 1.

### ExcessReturnLoss Function
The custom loss function, `ExcessReturnLoss`, is designed to maximize the Sharpe Ratio, which measures the performance of the trading signals relative to their risk. The loss function:
- **Calculates Excess Returns**: Based on the signals and the actual returns.
- **Computes the Sharpe Ratio**: As the mean excess return divided by the standard deviation of excess returns.
- **Negates the Sharpe Ratio**: So that minimizing the loss function maximizes the Sharpe Ratio.

By using this architecture and loss function, the model aims to generate trading signals that optimize returns relative to risk.

## Monte Carlo Simulation

Monte Carlo simulations are used to project the future performance of investment portfolios by running multiple scenarios based on historical data.

### Monte Carlo Simulation

1. Generate random variables  $Z \sim N(0, 1)$
2. Calculate the Cholesky decomposition of the covariance matrix $L$ 
3. Compute daily returns: $\text{dailyReturns} = \mu + L \cdot Z$

4. Compute portfolio values: $\text{portfolioValues} = \text{initialPortfolio} \cdot \prod_{t=1}^{T} (1 + \text{dailyReturns})$

5. Calculate performance metrics (e.g., Sharpe ratio, VaR, CVaR).




