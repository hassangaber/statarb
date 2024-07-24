# Quantitative Investing Dashboard

This repository contains a comprehensive dashboard for analyzing and visualizing the performance of various statistical arbitrage strategies over historical data. The dashboard supports strategies such as momentum, mean reversion, and volatility-based methods.

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

## Key Components

### Model and Loss Function

The dashboard uses a machine learning model to predict changes in stock returns. The model is trained on historical data and uses a custom loss function designed to optimize trading performance.

### Labeling Method

The data preparation process includes a sophisticated labeling method that categorizes historical price movements into buy, sell, or hold signals based on various technical indicators (and entropy features).

### Returns (naive targets) Model Architecture

The trading signal model uses a convolutional neural network (CNN) to capture temporal patterns in the stock data. The model is designed to generate optimal trading signals based on historical patterns.

### Monte Carlo Simulation

The dashboard includes a Monte Carlo simulation feature that projects future portfolio performance by running multiple scenarios based on historical data. This helps in assessing potential risks and returns of different investment strategies.

## Technical Details

For those interested in the technical aspects of the project, the full implementation includes detailed mathematical formulas and code snippets for:

- Calculating changes in returns and technical indicators
- Implementing the returns model architecture
- Defining the custom loss function
- Performing Monte Carlo simulations

These details have been omitted from this README for brevity, but can be found in the source code and accompanying documentation.