import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class MiscFeatures:
    data: pd.DataFrame

    def fit_transform(self) -> pd.DataFrame:
        df = self.data.copy()

        # Ensure the DataFrame is sorted by ['ID', 'DATE']
        df = df.sort_values(['ID', 'DATE']).reset_index(drop=True)

        # Group by 'ID' to perform feature engineering per ID
        df_grouped = df.groupby('ID')

        features = []

        for id, group in df_grouped:
            group_features = group.copy()

            # Price Range
            group_features['Price_Range'] = group_features['HIGH'] - group_features['CLOSE']
            """
            Price Range:
            - Measures the daily range of price movement.
            - High price ranges can indicate high volatility periods.
            """

            # Normalized Volume
            group_features['Normalized_Volume'] = group_features['VOLUME'] / group_features['VOLUME'].rolling(window=20).mean()
            """
            Normalized Volume:
            - Volume normalized by a moving average of the volume.
            - Helps identify unusual trading activity.
            """

            # Volatility Change
            group_features['LOG_VOLATILITY_90D'] = np.log(group_features['VOLATILITY_90D']/group_features['VOLATILITY_90D'].shift(1))
            """
            Volatility Change:
            - Measures the percentage change in volatility.
            - Useful for detecting sudden changes in market conditions.
            """

            # Price Momentum
            group_features['Price_Momentum'] = group_features['CLOSE'] - group_features['CLOSE'].shift(4)
            """
            Price Momentum:
            - Difference between the current close price and its value 10 days ago.
            - Useful for capturing short-term trends.
            """

            # Rolling Correlation
            group_features['Rolling_Correlation'] = group_features['CLOSE'].rolling(window=20).corr(group_features['VOLUME'])
            """
            Rolling Correlation:
            - Rolling correlation between close price and volume.
            - Useful for understanding the relationship between price and volume over time.
            """

            # Bollinger Bands
            sma_20 = group_features['CLOSE'].rolling(window=20).mean()
            std_20 = group_features['CLOSE'].rolling(window=20).std()
            group_features['Bollinger_Upper'] = sma_20 + (std_20 * 2)
            group_features['Bollinger_Lower'] = sma_20 - (std_20 * 2)
            """
            Bollinger Bands:
            - Upper and lower bands based on the 20-day SMA and standard deviation.
            - Useful for identifying overbought and oversold conditions.
            """

            # Relative Volatility Index (RVI)
            group_features['RVI'] = self.calculate_rvi(group_features['VOLATILITY_90D'], window=14)
            """
            Relative Volatility Index (RVI):
            - Similar to RSI but for volatility.
            - Helps identify periods of high and low volatility.
            """

            features.append(group_features)

        df_with_misc_features = pd.concat(features, ignore_index=True)
        
        return df_with_misc_features

    # Relative Volatility Index (RVI)
    def calculate_rvi(self, series: pd.Series, window: int) -> pd.Series:
        up_volatility = series.diff().apply(lambda x: x if x > 0 else 0)
        down_volatility = series.diff().apply(lambda x: -x if x < 0 else 0)
        up_mean = up_volatility.rolling(window=window).mean()
        down_mean = down_volatility.rolling(window=window).mean()
        rvi = 100 * up_mean / (up_mean + down_mean)
        return rvi