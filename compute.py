import pandas as pd

def calculate_cross_correlation(df, stock1, stock2, max_lag, start_date, end_date):
    """ Calculate cross-correlation between two stock time series for a range of lags within the given date range. """
    if stock1 not in df['ID'].unique() or stock2 not in df['ID'].unique():
        return pd.Series()  # Return an empty series if the stocks are not in the DataFrame

    # Filter data for each stock within the date range
    stock1_df = df[(df['ID'] == stock1) & (df.index >= start_date) & (df.index <= end_date)]['close']
    stock2_df = df[(df['ID'] == stock2) & (df.index >= start_date) & (df.index <= end_date)]['close']

    # Check that both dataframes have data
    if stock1_df.empty or stock2_df.empty:
        return pd.Series()

    # Compute cross-correlations for different lags
    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        shifted_df = stock2_df.shift(lag)  # Shift stock2 relative to stock1
        corr = stock1_df.corr(shifted_df)
        correlations[lag] = corr

    return pd.Series(correlations)

