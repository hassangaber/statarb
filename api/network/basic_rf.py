import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import onnxruntime as ort

def run_inference_onnx(model_path: str, input_data: np.ndarray) -> np.ndarray:
    """
    Run inference on an ONNX model.

    Parameters:
        model_path (str): Path to the ONNX model.
        input_data (np.ndarray): Input data for the model.

    Returns:
        np.ndarray: Model predictions.
    """
    sess = ort.InferenceSession(model_path)

    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)
    if len(input_data.shape) == 1:
        input_data = np.expand_dims(input_data, axis=0)

    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: input_data.astype(np.float32)})
    return result[0]

pd.options.display.float_format = "{:,.2f}".format

class PortfolioPrediction:
    def __init__(
        self,
        filename: str,
        stock_id: str,
        train_end_date: str,
        test_start_date: str,
        start_date: str,
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
        self.scaler = StandardScaler().set_output(transform='pandas')

        self.initial_investment = initial_investment
        self.share_volume = share_volume

    def preprocess_data(self) -> np.ndarray:
        """
        Preprocess the data for training and testing.

        Returns:
            np.ndarray: Test set features.
        """
        self.df = self.df[(self.df.ID == self.stock_id) & (self.df.DATE >= self.start_date)].sort_values('DATE')
        self.df.dropna(inplace=True)

        self.df["target"] = (self.df["RETURNS"] > 0).astype(int).values

        features = ["CLOSE", "VOLATILITY_90D", "VOLUME", "HIGH"]

        train_df = self.df[self.df["DATE"] <= self.train_end_date]
        test_df = self.df[self.df["DATE"] >= self.test_start_date]

        self.scaler.fit(train_df[features])
        train_df[features] = self.scaler.transform(train_df[features])
        test_df[features] = self.scaler.transform(test_df[features])

        self.X_train = train_df[features].values
        self.y_train = train_df["target"].values

        self.X_test = test_df[features].values
        self.y_test = test_df["target"].values

        self.volatility_train = train_df["VOLATILITY_90D"].values 

        return self.X_test

    def backtest(self) -> pd.DataFrame:
        """
        Backtest the model and calculate portfolio performance.

        Returns:
            pd.DataFrame: Portfolio performance metrics.
        """
        self.portfolio = (
            self.df[["DATE", "CLOSE", "RETURNS"]]
            .loc[self.df["DATE"] >= self.test_start_date]
            .copy()
        )
        self.portfolio.sort_values(by="DATE", inplace=True)
        self.portfolio.reset_index(drop=True, inplace=True)

        predicted_probabilities = run_inference_onnx('assets/model.onnx', self.X_test.astype(np.float32))

        self.portfolio["predicted_signal"] = (predicted_probabilities > 0.5)
        self.portfolio["p_buy"] = predicted_probabilities  # Probability of buy
        self.portfolio["p_sell"] = 1 - predicted_probabilities  # Probability of sell

        self.portfolio["portfolio_value"] = self.initial_investment
        self.portfolio["cumulative_shares"] = 0  # Initialize cumulative shares column
        self.portfolio["cumulative_share_cost"] = 0
        number_of_shares = 0
        cumulative_share_cost = 0  # Total cost of shares held

        for i, row in self.portfolio.iterrows():
            if i == 0:
                continue

            if (
                row["predicted_signal"] == 1
                and self.portfolio.at[i - 1, "portfolio_value"]
                > row["CLOSE"] * self.share_volume
            ):
                # Buy shares
                number_of_shares += self.share_volume
                purchase_cost = row["CLOSE"] * self.share_volume
                self.portfolio.at[i, "portfolio_value"] -= purchase_cost
                cumulative_share_cost += purchase_cost

            elif row["predicted_signal"] == 0 and number_of_shares >= self.share_volume:
                # Sell shares
                number_of_shares -= self.share_volume
                sale_proceeds = row["CLOSE"] * self.share_volume
                self.portfolio.at[i, "portfolio_value"] += sale_proceeds
                cumulative_share_cost -= row["CLOSE"] * self.share_volume

            # Update cumulative share and cost tracking
            self.portfolio.at[i, "cumulative_shares"] = number_of_shares
            self.portfolio.at[i, "cumulative_share_cost"] = cumulative_share_cost

            # Update portfolio value for current market value of shares
            if number_of_shares > 0:
                current_market_value = number_of_shares * row["CLOSE"]
                self.portfolio.at[i, "portfolio_value"] += (
                    current_market_value - cumulative_share_cost
                )

            # Calculate alpha as difference between portfolio return and benchmark return
            if i > 0:
                portfolio_daily_return = (self.portfolio.at[i, "portfolio_value"] / self.portfolio.at[0, "portfolio_value"] - 1)
                self.portfolio.at[i, "alpha"] = round(portfolio_daily_return, 3)  # - benchmark_daily_return

        return self.portfolio[["DATE",
                               "CLOSE",
                               "RETURNS",
                               "p_buy",
                               "p_sell",
                               "predicted_signal",
                               "cumulative_shares",
                               "cumulative_share_cost",
                               "portfolio_value",
                               "alpha"]]
