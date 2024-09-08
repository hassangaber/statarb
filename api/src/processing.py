#!/usr/bin/env/ python3.11

from enum import Enum
from typing import Final, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.stattools import adfuller


# defining features and their families
class Features(Enum):
    MACRO: list[str] = ['MACRO_INFLATION_EXPECTATION', 'MACRO_US_ECONOMY', 'MACRO_TREASURY_10Y', 'MACRO_TREASURY_5Y','MACRO_TREASURY_2Y','MACRO_VIX', 'MACRO_US_DOLLAR','MACRO_GOLD','MACRO_OIL']
    FUNDAMENTALS: list[str] = ['HIGH', 'VOLUME', 'VOLATILITY_90D']
    BETA: list[str] = ['BETA_TS']

ALL: Final[list[str]] = Features.MACRO+Features.FUNDAMENTALS+Features.BETA

class TimeSeriesPreprocessor(BaseEstimator, TransformerMixin):
    """
    A preprocessing class for time series data that performs necessary transformations
    before stationarity checks. This class is compatible with sklearn pipelines.
    """

    def __init__(self, 
                 fill_method: str = 'ffill',
                 outlier_method: str = 'iqr',
                 outlier_params: dict = {'lower': 0.01, 'upper': 0.99},
                 detrend_method: Optional[str] = None,
                 scaling_method: str = 'standardize'):
        """
        Initialize the TimeSeriesPreprocessor.

        Args:
            fill_method (str): Method to fill missing values ('ffill', 'bfill', or 'interpolate')
            outlier_method (str): Method to handle outliers ('iqr' or 'zscore')
            outlier_params (dict): Parameters for outlier detection
            detrend_method (Optional[str]): Method for detrending (None, 'difference', or 'polynomial')
            scaling_method (str): Method for scaling ('standardize', 'minmax', or 'robust')
        """
        self.fill_method = fill_method
        self.outlier_method = outlier_method
        self.outlier_params = outlier_params
        self.detrend_method = detrend_method
        self.scaling_method = scaling_method

    def fit(self, X: pd.Series, y=None) -> 'TimeSeriesPreprocessor':
        """
        Fit the preprocessor to the data. This method is required for sklearn compatibility
        but doesn't perform any operation in this case.

        Args:
            X (pd.Series): The input time series
            y: Ignored, present for sklearn compatibility

        Returns:
            self: The fitted preprocessor
        """
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """
        Apply the preprocessing steps to the input time series.

        Args:
            X (pd.Series): The input time series

        Returns:
            pd.Series: The preprocessed time series
        """
        # Fill missing values
        X = self._fill_missing(X)

        # Handle outliers
        X = self._handle_outliers(X)

        # Detrend if specified
        if self.detrend_method:
            X = self._detrend(X)

        # Scale the data
        X = self._scale(X)

        return X

    def _fill_missing(self, X: pd.Series) -> pd.Series:
        """Fill missing values in the time series."""
        if self.fill_method == 'ffill':
            return X.fillna(method='ffill')
        elif self.fill_method == 'bfill':
            return X.fillna(method='bfill')
        elif self.fill_method == 'interpolate':
            return X.interpolate()
        else:
            raise ValueError("Invalid fill method. Choose 'ffill', 'bfill', or 'interpolate'.")

    def _handle_outliers(self, X: pd.Series) -> pd.Series:
        """Handle outliers in the time series."""
        if self.outlier_method == 'iqr':
            q1, q3 = X.quantile([self.outlier_params['lower'], self.outlier_params['upper']])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return X.clip(lower_bound, upper_bound)
        elif self.outlier_method == 'zscore':
            z_scores = stats.zscore(X)
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3)
            return X[filtered_entries]
        else:
            raise ValueError("Invalid outlier method. Choose 'iqr' or 'zscore'.")

    def _detrend(self, X: pd.Series) -> pd.Series:
        """Detrend the time series."""
        if self.detrend_method == 'difference':
            return X.diff().dropna()
        elif self.detrend_method == 'polynomial':
            x = np.arange(len(X))
            coeffs = np.polyfit(x, X, deg=1)
            trend = np.polyval(coeffs, x)
            return X - trend
        else:
            raise ValueError("Invalid detrend method. Choose 'difference' or 'polynomial'.")

    def _scale(self, X: pd.Series) -> pd.Series:
        """Scale the time series."""
        if self.scaling_method == 'standardize':
            return (X - X.mean()) / X.std()
        elif self.scaling_method == 'minmax':
            return (X - X.min()) / (X.max() - X.min())
        elif self.scaling_method == 'robust':
            median = X.median()
            q1, q3 = X.quantile([0.25, 0.75])
            iqr = q3 - q1
            return (X - median) / iqr
        else:
            raise ValueError("Invalid scaling method. Choose 'standardize', 'minmax', or 'robust'.")


class FractionalDifferencing(BaseEstimator, TransformerMixin):
    """
    A class for applying fractional differencing to a time series with maximum
    memory preservation, following the methodology in "Advances in Financial
    Machine Learning" by Marcos López de Prado.
    """

    def __init__(self, d: Optional[float] = None, threshold: float = 1e-5):
        """
        Initialize the FractionalDifferencing transformer.

        Args:
            d (Optional[float]): The differencing parameter. If None, it will be estimated.
            threshold (float): The threshold for the cumulative weight in memory preservation.
        """
        self.d = d
        self.threshold = threshold

    def fit(self, X: pd.Series, y=None) -> 'FractionalDifferencing':
        """
        Fit the fractional differencing parameter if not provided.

        Args:
            X (pd.Series): The input time series
            y: Ignored, present for sklearn compatibility

        Returns:
            self: The fitted transformer
        """
        if self.d is None:
            self.d = self._estimate_d(X)
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        """
        Apply fractional differencing to the time series.

        Args:
            X (pd.Series): The input time series

        Returns:
            pd.Series: The fractionally differenced time series
        """
        weights = self._get_weights()
        width = len(weights) - 1
        output = []
        for i in range(width, len(X)):
            output.append(np.dot(weights, X.iloc[i-width:i+1]))
        return pd.Series(output, index=X.index[width:])

    def _get_weights(self) -> np.ndarray:
        """Calculate the weights for fractional differencing."""
        weights = [1.]
        k = 1
        while True:
            w = -weights[-1] * (self.d - k + 1) / k
            if abs(w) < self.threshold:
                break
            weights.append(w)
            k += 1
        return np.array(weights[::-1])

    def _estimate_d(self, X: pd.Series) -> float:
        """
        Estimate the optimal differencing parameter d using ADF test.

        This method tries different values of d and selects the one that
        makes the series stationary while preserving maximum memory.
        """
        d_range = np.linspace(0, 1, 100)
        results = []
        for d in d_range:
            temp_transformer = FractionalDifferencing(d=d, threshold=self.threshold)
            diff_series = temp_transformer.fit_transform(X)
            adf_result = adfuller(diff_series, maxlag=1, regression='c', autolag=None)
            results.append((d, adf_result[0], adf_result[1]))
        
        results_df = pd.DataFrame(results, columns=['d', 'adf_stat', 'p_value'])
        stationary_results = results_df[results_df['p_value'] < 0.05]
        
        if stationary_results.empty:
            return 1  # If no stationary result, return maximum differencing
        else:
            return stationary_results.iloc[-1]['d']  # Return the highest d that makes the series stationary
