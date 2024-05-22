import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

from api.network.inference import run_inference_onnx
from api.src.portfolio_sim import update_portfolio_rf, update_portfolio_new, update_portfolio_softmax

pd.options.display.float_format = "{:,.2f}".format


class PortfolioPrediction:
    def __init__(
        self,
        filename: str,
        stock_id: str,
        test_start_date: str,
        train_end_date: str = "2024-01-01",
        start_date: str = "2015-01-01",
        initial_investment: int = 10000,
        share_volume: int = 5,
    ):
        """
        Initialize PortfolioPrediction object.

        Parameters:
            filename (str): Path to CSV file with stock data.
            stock_id (str): Stock ID to filter the data.
            train_end_date (str): End date for the training period.
            test_start_date (str): Start date for the test period.
            start_date (str): Start date for filtering the data.
            initial_investment (int): Initial investment amount.
            share_volume (int): Number of shares to buy/sell per transaction.
        """
        self.df = pd.read_csv(filename)
        self.df["DATE"] = pd.to_datetime(self.df["DATE"])
        self.df.sort_values("DATE", inplace=True)

        self.stock_id = stock_id
        self.train_end_date = pd.to_datetime(train_end_date)
        self.test_start_date = pd.to_datetime(test_start_date)
        self.start_date = pd.to_datetime(start_date)

        self.model = None
        self.train_loader = None
        self.portfolio = pd.DataFrame()
        self.scaler = StandardScaler().set_output(transform="pandas")

        self.initial_investment = initial_investment
        self.share_volume = share_volume

    def preprocess_test_data(self) -> np.ndarray:
        """
        Preprocess the data for training and testing.

        Returns:
            np.ndarray: Test set features.
        """
        self.df = self.df[(self.df.ID == self.stock_id) & (self.df.DATE >= self.start_date)].sort_values("DATE")
        self.df.dropna(inplace=True)

        features = [
            "CLOSE",
            "VOLUME",
            "HIGH",
            "RETURNS",
            "VOLATILITY_90D",
            "CLOSE_SMA_3D",
            "CLOSE_EWMA_3D",
            "VOLATILITY_90D_SMA_3D",
            "VOLATILITY_90D_EWMA_3D",
            "CLOSE_SMA_9D",
            "CLOSE_EWMA_9D",
            "VOLATILITY_90D_SMA_9D",
            "VOLATILITY_90D_EWMA_9D",
            "CLOSE_SMA_21D",
            "CLOSE_EWMA_21D",
            "VOLATILITY_90D_SMA_21D",
            "VOLATILITY_90D_EWMA_21D",
            "CLOSE_SMA_50D",
            "CLOSE_EWMA_50D",
            "VOLATILITY_90D_SMA_50D",
            "VOLATILITY_90D_EWMA_50D",
            "CLOSE_SMA_65D",
            "CLOSE_EWMA_65D",
            "VOLATILITY_90D_SMA_65D",
            "VOLATILITY_90D_EWMA_65D",
            "CLOSE_SMA_120D",
            "CLOSE_EWMA_120D",
            "VOLATILITY_90D_SMA_120D",
            "VOLATILITY_90D_EWMA_120D",
            "CLOSE_SMA_360D",
            "CLOSE_EWMA_360D",
            "VOLATILITY_90D_SMA_360D",
            "VOLATILITY_90D_EWMA_360D",
            "CLOSE_ROC_3D",
            "HIGH_ROC_3D",
            "VOLATILITY_90D_ROC_3D",
            "CLOSE_ROC_9D",
            "HIGH_ROC_9D",
            "VOLATILITY_90D_ROC_9D",
            "CLOSE_ROC_21D",
            "HIGH_ROC_21D",
            "VOLATILITY_90D_ROC_21D",
            "CLOSE_ROC_50D",
            "HIGH_ROC_50D",
            "VOLATILITY_90D_ROC_50D",
            "CLOSE_ROC_65D",
            "HIGH_ROC_65D",
            "VOLATILITY_90D_ROC_65D",
            "CLOSE_ROC_120D",
            "HIGH_ROC_120D",
            "VOLATILITY_90D_ROC_120D",
            "CLOSE_ROC_360D",
            "HIGH_ROC_360D",
            "VOLATILITY_90D_ROC_360D",
        ]

        train_df = self.df[self.df["DATE"] < self.train_end_date]
        test_df = self.df[self.df["DATE"] >= self.test_start_date]

        train_df[features] = self.scaler.fit_transform(train_df[features])
        test_df[features] = self.scaler.transform(test_df[features])

        self.X_test = test_df[features].values

        return self.X_test

    def backtest(self, model_id: str = "model") -> pd.DataFrame:
        """
        Backtest the model and calculate portfolio performance.

        Returns:
            pd.DataFrame: Portfolio performance metrics.
        """
        self.portfolio = self.df[["DATE", "CLOSE", "RETURNS"]].loc[self.df["DATE"] >= self.test_start_date].copy()

        self.portfolio.sort_values(by="DATE", inplace=True)
        self.portfolio.reset_index(drop=True, inplace=True)

        Pr = run_inference_onnx(f"assets/{model_id}.onnx", self.X_test.astype(np.float32))

        self.portfolio["predicted_signal"] = [p for p in Pr]

        self.portfolio['p_buy'] = [p[2] for p in Pr]
        self.portfolio['p_hold'] = [p[1] for p in Pr]
        self.portfolio['p_sell'] = [p[0] for p in Pr]

        self.portfolio["graph_signal"] = [np.argmax(p) for p in Pr]

        # Initialize self.portfolio columns
        self.portfolio["cash_on_hand"] = self.initial_investment
        self.portfolio["share_value"] = 0
        self.portfolio["total_portfolio_value"] = self.initial_investment
        self.portfolio["cumulative_shares"] = 0
        self.portfolio["PnL"] = 0.0
        self.portfolio["position"] = "no position"

        # Iterate through the DataFrame to update each row
        for i in range(1, len(self.portfolio)):
            self.portfolio.iloc[i] = update_portfolio_softmax(
                self.portfolio.iloc[i], self.portfolio.iloc[i - 1], self.initial_investment, self.share_volume, tr=0.45
            )

        # Return the relevant columns
        return self.portfolio[
            [
                "DATE",
                "CLOSE",
                "RETURNS",
                "cash_on_hand",
                "share_value",
                "total_portfolio_value",
                "position",
                "cumulative_shares",
                "PnL",
                "predicted_signal",
                'p_sell',
                'p_hold',
                'p_buy',
                'graph_signal'
            ]
        ]


class PortfolioPredictionRF:
    def __init__(
        self,
        filename: str,
        stock_id: str,
        test_start_date: str,
        model_path: str,
        train_end_date: str = "2024-01-01",
        start_date: str = "2015-01-01",
        initial_investment: int = 10000,
        share_volume: int = 5,
    ):
        """
        Initialize PortfolioPredictionRF object.

        Parameters:
            filename (str): Path to CSV file with stock data.
            stock_id (str): Stock ID to filter the data.
            train_end_date (str): End date for the training period.
            test_start_date (str): Start date for the test period.
            start_date (str): Start date for filtering the data.
            initial_investment (int): Initial investment amount.
            share_volume (int): Number of shares to buy/sell per transaction.
            model_path (str): Path to the pickled model file.
        """
        self.df = pd.read_csv(filename)
        self.df["DATE"] = pd.to_datetime(self.df["DATE"])
        self.df.sort_values("DATE", inplace=True)

        self.stock_id = stock_id
        self.train_end_date = pd.to_datetime(train_end_date)
        self.test_start_date = pd.to_datetime(test_start_date)
        self.start_date = pd.to_datetime(start_date)

        self.model = self.load_model(model_path)
        self.portfolio = pd.DataFrame()
        self.scaler = StandardScaler().set_output(transform="pandas")

        self.initial_investment = initial_investment
        self.share_volume = share_volume

    def load_model(self, model_path: str = "assets/xgb.pkl"):
        """
        Load a model from a pickle file.

        Parameters:
            model_path (str): Path to the pickle file containing the model.

        Returns:
            model: The loaded model.
        """
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        return model

    def preprocess_test_data(self) -> np.ndarray:
        """
        Preprocess the data for training and testing.

        Returns:
            np.ndarray: Test set features.
        """
        self.df = self.df[(self.df.ID == self.stock_id) & (self.df.DATE >= self.start_date)].sort_values("DATE")
        self.df.dropna(inplace=True)

        self.df["label"] = (self.df["RETURNS"] > 0).astype(int).values

        features = [
            "CLOSE",
            "VOLUME",
            "HIGH",
            "RETURNS",
            "VOLATILITY_90D",
            "CLOSE_SMA_3D",
            "CLOSE_EWMA_3D",
            "VOLATILITY_90D_SMA_3D",
            "VOLATILITY_90D_EWMA_3D",
            "CLOSE_SMA_9D",
            "CLOSE_EWMA_9D",
            "VOLATILITY_90D_SMA_9D",
            "VOLATILITY_90D_EWMA_9D",
            "CLOSE_SMA_21D",
            "CLOSE_EWMA_21D",
            "VOLATILITY_90D_SMA_21D",
            "VOLATILITY_90D_EWMA_21D",
            "CLOSE_SMA_50D",
            "CLOSE_EWMA_50D",
            "VOLATILITY_90D_SMA_50D",
            "VOLATILITY_90D_EWMA_50D",
            "CLOSE_SMA_65D",
            "CLOSE_EWMA_65D",
            "VOLATILITY_90D_SMA_65D",
            "VOLATILITY_90D_EWMA_65D",
            "CLOSE_SMA_120D",
            "CLOSE_EWMA_120D",
            "VOLATILITY_90D_SMA_120D",
            "VOLATILITY_90D_EWMA_120D",
            "CLOSE_SMA_360D",
            "CLOSE_EWMA_360D",
            "VOLATILITY_90D_SMA_360D",
            "VOLATILITY_90D_EWMA_360D",
            "CLOSE_ROC_3D",
            "HIGH_ROC_3D",
            "VOLATILITY_90D_ROC_3D",
            "CLOSE_ROC_9D",
            "HIGH_ROC_9D",
            "VOLATILITY_90D_ROC_9D",
            "CLOSE_ROC_21D",
            "HIGH_ROC_21D",
            "VOLATILITY_90D_ROC_21D",
            "CLOSE_ROC_50D",
            "HIGH_ROC_50D",
            "VOLATILITY_90D_ROC_50D",
            "CLOSE_ROC_65D",
            "HIGH_ROC_65D",
            "VOLATILITY_90D_ROC_65D",
            "CLOSE_ROC_120D",
            "HIGH_ROC_120D",
            "VOLATILITY_90D_ROC_120D",
            "CLOSE_ROC_360D",
            "HIGH_ROC_360D",
            "VOLATILITY_90D_ROC_360D",
        ]

        test_df = self.df[self.df["DATE"] >= self.test_start_date]
        test_df[features] = self.scaler.fit_transform(test_df[features])

        self.X_test = test_df[features].values
        self.y_test = test_df["label"].values

        return self.X_test

    def backtest(self, buy_probability_threshold: float = 0.5) -> pd.DataFrame:
        """
        Backtest the model and calculate portfolio performance.

        Returns:
            pd.DataFrame: Portfolio performance metrics.
        """
        self.portfolio = self.df[["DATE", "CLOSE", "RETURNS"]].loc[self.df["DATE"] >= self.test_start_date].copy()

        self.portfolio.sort_values(by="DATE", inplace=True)
        self.portfolio.reset_index(drop=True, inplace=True)

        Pr = self.model.predict(self.X_test)

        self.portfolio["predicted_signal"] = Pr
        print(self.portfolio.predicted_signal)

        # Initialize self.portfolio columns
        self.portfolio["cash_on_hand"] = self.initial_investment
        self.portfolio["share_value"] = 0
        self.portfolio["total_portfolio_value"] = self.initial_investment
        self.portfolio["cumulative_shares"] = 0
        self.portfolio["PnL"] = 0.0
        self.portfolio["position"] = "no position"

        # Iterate through the DataFrame to update each row
        for i in range(1, len(self.portfolio)):
            self.portfolio.iloc[i] = update_portfolio_rf(
                self.portfolio.iloc[i], self.portfolio.iloc[i - 1], self.initial_investment, self.share_volume
            )

        # Return the relevant columns
        return self.portfolio[
            [
                "DATE",
                "CLOSE",
                "RETURNS",
                "cash_on_hand",
                "share_value",
                "total_portfolio_value",
                "position",
                "cumulative_shares",
                "PnL",
                "predicted_signal",
            ]
        ]
