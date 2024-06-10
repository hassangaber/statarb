from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class TechnicalIndicators:
    data: pd.DataFrame
    sma_days: List[int] = field(default_factory=lambda: [3, 9, 15, 21, 60, 120])
    ewma_days: List[int] = field(default_factory=lambda: [3, 9, 15, 21, 60, 120])
    rsi_days: List[int] = field(default_factory=lambda: [4, 8, 15])
    roc_days: List[int] = field(default_factory=lambda: [4, 8, 15, 21])
    lag_days: List[int] = field(default_factory=lambda: [1, 4, 8, 21])
    return_days: List[int] = field(default_factory=lambda: [4, 8, 21])

    def fit_transform(self) -> pd.DataFrame:
        df = self.data.copy()

        # Check if the DataFrame is sorted by ['ID', 'DATE']
        if not df.sort_values(['ID', 'DATE']).equals(df):
            df = df.sort_values(['ID', 'DATE']).reset_index(drop=True)

        # Group by 'ID' to ensure features are created separately for each ID
        df_grouped = df.groupby('ID')

        features = []

        for _, group in tqdm(df_grouped):
            group_features = group.copy()
            
            # Create SMAs
            for days in self.sma_days:
                for col in ['CLOSE', 'HIGH', 'VOLUME', 'VOLATILITY_90D']:
                    group_features[f'{col}_SMA_{days}'] = self.calculate_sma(group_features[col], days)

            # Create EWMAs
            for days in self.ewma_days:
                group_features[f'CLOSE_EWMA_{days}'] = self.calculate_ewma(group_features['CLOSE'], days)

            # Create RSIs
            for days in self.rsi_days:
                group_features[f'CLOSE_RSI_{days}'] = self.calculate_rsi(group_features['CLOSE'], days)

            # Create ROCs
            for days in self.roc_days:
                for col in ['CLOSE', 'HIGH', 'VOLUME', 'VOLATILITY_90D']:
                    group_features[f'{col}_ROC_{days}'] = self.calculate_roc(group_features[col], days)

            # Create lagged features
            for days in self.lag_days:
                for col in ['CLOSE', 'HIGH', 'VOLUME', 'VOLATILITY_90D']:
                    group_features[f'{col}_LAG_{days}'] = self.calculate_lag(group_features[col], days)

            # Calculate log returns
            group_features['LOG_RETURN'] = self.calculate_log_returns(group_features['CLOSE'])


            for days in self.return_days:
                group_features[f'LOG_RETURN_ROC_{days}'] = self.calculate_roc(group_features['LOG_RETURN'], days)
                group_features[f'LOG_RETURN_RSI_{days}'] = self.calculate_rsi(group_features['LOG_RETURN'], days)

            features.append(group_features)

        df_with_features = pd.concat(features, ignore_index=True)
        df_with_features = df_with_features.dropna().reset_index(drop=True)

        return df_with_features

    # Simple Moving Average (SMA)
    def calculate_sma(self, series: pd.Series, window: int) -> pd.Series:
        """
        SMA_t = (P_t + P_{t-1} + ... + P_{t-window+1}) / window
        """
        return series.rolling(window=window).mean()

    # Exponential Weighted Moving Average (EWMA)
    def calculate_ewma(self, series: pd.Series, span: int) -> pd.Series:
        """
        EWMA_t = α * P_t + (1 - α) * EWMA_{t-1}
        where α = 2 / (span + 1)
        """
        return series.ewm(span=span, adjust=False).mean()

    # Relative Strength Index (RSI)
    def calculate_rsi(self, series: pd.Series, window: int) -> pd.Series:
        """
        RSI = 100 - (100 / (1 + RS))
        where RS = average_gain / average_loss
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Rate of Change (ROC)
    def calculate_roc(self, series: pd.Series, window: int) -> pd.Series:
        """
        ROC = (P_t - P_{t-window}) / P_{t-window}
        """
        return series.pct_change(periods=window, fill_method=None)

    # Lagged Features
    def calculate_lag(self, series: pd.Series, lag: int) -> pd.Series:
        """
        Lag_t = P_{t-lag}
        """
        return series.shift(lag)

    # Log Returns
    def calculate_log_returns(self, series: pd.Series) -> pd.Series:
        """
        Log Return_t = log(P_t / P_{t-1})
        """
        return np.log(series / series.shift(1))
    
