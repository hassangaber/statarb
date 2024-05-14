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
        test_start_date: str,
        train_end_date: str='2024-01-01',
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

        features = ["CLOSE", "VOLATILITY_90D", "VOLUME", "HIGH", "CLOSE_ROC_3D", "CLOSE_EWMA_120D"]

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
        self.portfolio["predicted_signal"] = (predicted_probabilities > 0.5).astype(int)
        self.portfolio["p_buy"] = predicted_probabilities
        self.portfolio["p_sell"] = 1 - predicted_probabilities

        # Initialize self.portfolio columns
        self.portfolio["cash_on_hand"] = self.initial_investment
        self.portfolio["share_value"] = 0
        self.portfolio["total_portfolio_value"] = self.initial_investment
        self.portfolio["cumulative_shares"] = 0
        self.portfolio["PnL"] = 0.0
        self.portfolio["position"] = "no position"

        def update_portfolio(row, previous_row):
            close_price = row["CLOSE"]
            signal = row["predicted_signal"]
            probability = row["p_buy"]
            
            # Hold if probability is between 45% and 55%
            if 0.45 <= probability <= 0.55:
                row["cumulative_shares"] = previous_row["cumulative_shares"]
                row["cash_on_hand"] = previous_row["cash_on_hand"]
                row["share_value"] = row["cumulative_shares"] * close_price
                row["total_portfolio_value"] = row["cash_on_hand"] + row["share_value"]
                row["PnL"] = row["total_portfolio_value"] - self.initial_investment
                row["position"] = "hold"
                return row
            
            if signal:  # Buy signal
                shares_bought = min(self.share_volume, row["cash_on_hand"] // close_price)
                if shares_bought > 0:
                    row["cumulative_shares"] = previous_row["cumulative_shares"] + shares_bought
                    row["cash_on_hand"] = previous_row["cash_on_hand"] - shares_bought * close_price
                    row["share_value"] = row["cumulative_shares"] * close_price
                    row["total_portfolio_value"] = row["cash_on_hand"] + row["share_value"]
                    row["PnL"] = row["total_portfolio_value"] - self.initial_investment
                    row["position"] = "buy"
                else:
                    row["position"] = "hold"
            
            else:  # Sell signal
                shares_sold = min(self.share_volume, previous_row["cumulative_shares"])
                if shares_sold > 0:
                    row["cumulative_shares"] = previous_row["cumulative_shares"] - shares_sold
                    row["cash_on_hand"] = previous_row["cash_on_hand"] + shares_sold * close_price
                    row["share_value"] = row["cumulative_shares"] * close_price
                    row["total_portfolio_value"] = row["cash_on_hand"] + row["share_value"]
                    row["PnL"] = row["total_portfolio_value"] - self.initial_investment
                    row["position"] = "sell"
                else:
                    row["position"] = "hold"

            # Update position to hold if no shares bought or sold
            if row["cumulative_shares"] == previous_row["cumulative_shares"]:
                row["position"] = "hold"
            
            return row

        # Initialize the first row based on initial investment
        self.portfolio.iloc[0] = update_portfolio(self.portfolio.iloc[0], self.portfolio.iloc[0])

        # Iterate through the DataFrame to update each row
        for i in range(1, len(self.portfolio)):
            self.portfolio.iloc[i] = update_portfolio(self.portfolio.iloc[i], self.portfolio.iloc[i - 1])

        print(self.portfolio)

        # Return the relevant columns
        result = self.portfolio[['DATE', 'CLOSE', 'cash_on_hand', 'share_value', "total_portfolio_value", "position", "cumulative_shares", "PnL", "predicted_signal"]]
        print(result)

        return result
