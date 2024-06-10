from dataclasses import dataclass
import pandas as pd
from scipy.stats import entropy as shannon_entropy
import antropy as ant  # type: ignore
from tqdm import tqdm

@dataclass
class EntropyFeatures:
    data: pd.DataFrame
    dropna: bool = True 

    def fit_transform(self) -> pd.DataFrame:
        df = self.data.copy()

        # Ensure the DataFrame is sorted by ['ID', 'DATE'] to preform transform on vector
        df = df.sort_values(['ID', 'DATE']).reset_index(drop=True)
        df_grouped = df.groupby('ID')

        features = []

        for _, group in tqdm(df_grouped):
            group_features = group.copy()

            # Calculate entropy features for different horizons
            horizons = [4, 8, 21, 120]
            for horizon in horizons:
                group_features[f'Shannon_Entropy_CLOSE_{horizon}'] = self.calculate_shannon_entropy(group_features['CLOSE'], horizon)
                group_features[f'Shannon_Entropy_VOLUME_{horizon}'] = self.calculate_shannon_entropy(group_features['VOLUME'], horizon)
                group_features[f'Permutation_Entropy_CLOSE_{horizon}'] = self.calculate_permutation_entropy(group_features['CLOSE'], horizon)
                group_features[f'Shannon_Entropy_HIGH_{horizon}'] = self.calculate_shannon_entropy(group_features['HIGH'], horizon)
                group_features[f'Permutation_Entropy_HIGH_{horizon}'] = self.calculate_permutation_entropy(group_features['HIGH'], horizon)

            features.append(group_features)

        df_with_entropy = pd.concat(features, ignore_index=True)
        
        if self.dropna:
            df_with_entropy = df_with_entropy.dropna()

        return df_with_entropy

    # Shannon Entropy
    def calculate_shannon_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """
        Shannon Entropy:
        - Measures the uncertainty or randomness in the data.
        - Higher entropy values indicate more unpredictability.
        - Useful for identifying periods of high market volatility or irregular trading activity.
        """
        entropy_values = series.rolling(window=window).apply(lambda x: shannon_entropy(pd.Series(x).value_counts()), raw=True)
        return entropy_values

    # Permutation Entropy
    def calculate_permutation_entropy(self, series: pd.Series, window: int, order: int = 3, delay: int = 1) -> pd.Series:
        """
        Permutation Entropy:
        - Measures the complexity of the time series.
        - Useful for capturing temporal dynamics and patterns.
        - Can help identify periods of regular and predictable market behavior.
        """
        entropy_values = series.rolling(window=window).apply(lambda x: ant.perm_entropy(x, order=order, delay=delay, normalize=True), raw=True)
        return entropy_values
