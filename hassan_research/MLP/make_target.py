import pandas as pd
import numpy as np

class TargetGenerator:
    def __init__(self, data: pd.DataFrame, horizon: int = 1, return_threshold: float = 0.01):
        """
        Args:
            data (pd.DataFrame): The dataframe containing the features.
            horizon (int): The number of periods to look ahead to determine the target.
            return_threshold (float): The threshold for the return to consider a significant price movement.
        """
        self.data = data.sort_values(['ID', 'DATE']).reset_index(drop=True)
        self.horizon = horizon
        self.return_threshold = return_threshold

    def generate_targets(self) -> pd.DataFrame:
        """
        Generate targets based on price direction and market instability.
        The target can be:
        - 1 if the price goes up significantly.
        - 0 if the price goes down significantly.
        - 2 if the market is considered unstable.
        
        Returns:
            pd.DataFrame: DataFrame with the target column added.
        """
        # Calculate future returns
        self.data['FUTURE_RETURN'] = self.data.groupby('ID')['CLOSE'].shift(-self.horizon) / self.data['CLOSE'] - 1
        
        # Define the direction target
        self.data['DIRECTION_TARGET'] = np.where(self.data['FUTURE_RETURN'] > self.return_threshold, 1, 
                                                 np.where(self.data['FUTURE_RETURN'] < -self.return_threshold, 0, np.nan))
        
        # Define the instability target based on volatility and entropy
        volatility_threshold = self.data['VOLATILITY_90D'].quantile(0.75)
        entropy_threshold = self.data['Shannon_Entropy_CLOSE_21'].quantile(0.75)
        
        self.data['INSTABILITY_TARGET'] = np.where((self.data['VOLATILITY_90D'] > volatility_threshold) & 
                                                   (self.data['Shannon_Entropy_CLOSE_21'] > entropy_threshold), 2, np.nan)
        
        # Combine the targets
        self.data['TARGET'] = np.where(~self.data['INSTABILITY_TARGET'].isna(), 
                                       self.data['INSTABILITY_TARGET'], 
                                       self.data['DIRECTION_TARGET'])
        
        # Drop rows where the target could not be calculated
        self.data = self.data.dropna(subset=['TARGET'])
        self.data['TARGET'] = self.data['TARGET'].astype(int)
        
        # Drop the intermediate columns used for target calculation
        self.data.drop(columns=['FUTURE_RETURN', 'DIRECTION_TARGET', 'INSTABILITY_TARGET'], inplace=True)
        
        return self.data
    