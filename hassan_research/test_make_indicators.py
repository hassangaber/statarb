import unittest

import numpy as np
import pandas as pd
from make_indicators import TechnicalIndicators


class TestTechnicalIndicators(unittest.TestCase):

    def setUp(self):
        # Create a simple DataFrame for testing
        data = {
            'ID': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'DATE': pd.date_range(start='1/1/2020', periods=10),
            'CLOSE': [10, 11, 12, 11, 10, 11, 12, 11, 10, 9],
            'HIGH': [10, 11, 13, 12, 11, 12, 13, 12, 11, 10],
            'VOLUME': [1000, 1100, 1200, 1100, 1000, 1100, 1200, 1100, 1000, 900],
            'VOLATILITY_90D': [0.1, 0.12, 0.11, 0.13, 0.12, 0.11, 0.1, 0.12, 0.13, 0.14]
        }
        self.df = pd.DataFrame(data)
        self.ti = TechnicalIndicators(data=self.df)

    def test_calculate_sma(self):
        sma_3 = self.ti.calculate_sma(self.df['CLOSE'], 3)
        expected_sma_3 = pd.Series([np.nan, np.nan, 11, 11.3333, 11, 10.6667, 11, 11.3333, 11, 10], name='CLOSE').round(4)
        pd.testing.assert_series_equal(sma_3.round(4), expected_sma_3, check_names=False)

    def test_calculate_ewma(self):
        ewma_3 = self.ti.calculate_ewma(self.df['CLOSE'], 3)
        expected_ewma_3 = pd.Series([10, 10.5, 11.25, 11.125, 10.5625, 10.7813, 11.3906, 11.1953, 10.5977, 9.7988], name='CLOSE').round(4)
        pd.testing.assert_series_equal(ewma_3.round(4), expected_ewma_3, check_names=False)

    def test_calculate_rsi(self):
        rsi_3 = self.ti.calculate_rsi(self.df['CLOSE'], 3)
        expected_rsi_3 = pd.Series([np.nan, np.nan, np.nan, 100.0, 0.0, 50.0, 100.0, 50.0, 0.0, 0.0], name='CLOSE').round(4)
        pd.testing.assert_series_equal(rsi_3.round(4), expected_rsi_3, check_names=False)

    def test_calculate_roc(self):
        roc_3 = self.ti.calculate_roc(self.df['CLOSE'], 3)
        expected_roc_3 = pd.Series([np.nan, np.nan, np.nan, 10.0, -16.6667, -8.3333, 0.0, -8.3333, -16.6667, -25.0], name='CLOSE').round(4)
        pd.testing.assert_series_equal(roc_3.round(4), expected_roc_3, check_names=False)

    def test_calculate_lag(self):
        lag_1 = self.ti.calculate_lag(self.df['CLOSE'], 1)
        expected_lag_1 = pd.Series([np.nan, 10, 11, 12, 11, 10, 11, 12, 11, 10], name='CLOSE')
        pd.testing.assert_series_equal(lag_1, expected_lag_1, check_names=False)

    def test_calculate_log_returns(self):
        log_returns = self.ti.calculate_log_returns(self.df['CLOSE'])
        expected_log_returns = pd.Series([np.nan, 0.0953, 0.0870, -0.0870, -0.0953, 0.0953, 0.0870, -0.0870, -0.0953, -0.1054], name='CLOSE').round(4)
        pd.testing.assert_series_equal(log_returns.round(4), expected_log_returns, check_names=False)
