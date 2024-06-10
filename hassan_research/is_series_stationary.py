import pandas as pd
import numpy as np
from typing import Tuple, Dict
from statsmodels.tsa.stattools import adfuller # type: ignore
from tqdm import tqdm

def check_stationarity(series: pd.Series, alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Check the stationarity of a time series using the Augmented Dickey-Fuller (ADF) test.
    
    Args:
        series (pd.Series): The time series data.
        alpha (float): Significance level for the ADF test.
    
    Returns:
        Tuple[bool, float]: A tuple containing a boolean indicating if the series is stationary and the p-value of the ADF test.
    """
    if len(series.dropna())==0:
        return None, None
    adf_result = adfuller(series.dropna())
    p_value = adf_result[1]
    is_stationary = p_value < alpha
    return is_stationary, p_value

def suggest_stationarity_methods(series: pd.Series) -> Dict[str, pd.Series]:
    """
    Suggest methods to make a non-stationary time series stationary.
    
    Args:
        series (pd.Series): The time series data.
    
    Returns:
        Dict[str, pd.Series]: A dictionary with suggested methods and their corresponding transformed series.
    """
    methods = {}
    
    # Differencing
    diff_series = series.diff().dropna()
    methods['Differencing'] = diff_series
    
    # Log Transformation (only for positive values)
    if (series > 0).all():
        log_series = np.log(series).dropna()
        methods['Log Transformation'] = log_series
    
    # Detrending
    trend = np.polyfit(range(len(series)), series, 1)
    detrended_series = series - (trend[0] * range(len(series)) + trend[1])
    methods['Detrending'] = detrended_series
    
    return methods

def process_series(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Process the series to check for stationarity and apply necessary transformations.
    
    Args:
        df (pd.DataFrame): The DataFrame containing time series data.
        alpha (float): Significance level for the ADF test.
    
    Returns:
        pd.DataFrame: A DataFrame with transformed series.
    """
    processed_df = df.copy()

    # Ensure the DataFrame is sorted by ['ID', 'DATE']
    processed_df = processed_df.sort_values(['ID', 'DATE']).reset_index(drop=True)

    # Group by 'ID' to perform stationarity checks and transformations per ID
    df_grouped = processed_df.groupby('ID')

    features = []

    for id, group in tqdm(df_grouped):
        group_processed = group.copy()

        for col in ['HIGH', 'VOLUME', 'VOLATILITY_90D']:
            is_stationary, p_value = check_stationarity(group_processed[col], alpha)
            print(f"ID {id} - {col}: Stationary={is_stationary}, p-value={p_value}")
            if not is_stationary and is_stationary is not None:
                methods = suggest_stationarity_methods(group_processed[col])
                # Choose differencing by default for simplicity, but other methods can be selected based on the context
                group_processed[col] = methods['Log Transformation']
                # Recheck stationarity after transformation
                is_stationary, p_value = check_stationarity(group_processed[col], alpha)
                print(f"ID {id} - {col} after differencing: Stationary={is_stationary}, p-value={p_value}")
            if is_stationary is None:
                group_processed[col] = group_processed[col]
        
        features.append(group_processed)

    df_with_features = pd.concat(features, ignore_index=True)
    
    return df_with_features

