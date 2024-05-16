import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from api.network.inference import run_inference_onnx
from api.src.portfolio_sim import update_portfolio

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

    def preprocess_test_data(self) -> np.ndarray:
        """
        Preprocess the data for training and testing.

        Returns:
            np.ndarray: Test set features.
        """
        self.df = self.df[(self.df.ID == self.stock_id) & (self.df.DATE >= self.start_date)].sort_values('DATE')
        self.df.dropna(inplace=True)

        self.df["target"] = (self.df["RETURNS"] > 0).astype(int).values

        features = ["CLOSE", "VOLATILITY_90D", "VOLUME", "HIGH", "CLOSE_ROC_3D", "CLOSE_EWMA_120D"]

        #train_df = self.df[self.df["DATE"] <= self.train_end_date]
        test_df = self.df[self.df["DATE"] >= self.test_start_date]

        #self.scaler.fit(train_df[features])
        # train_df[features] = self.scaler.transform(train_df[features])
        test_df[features] = self.scaler.fit_transform(test_df[features])

        #self.X_train = train_df[features].values
        #self.y_train = train_df["target"].values

        self.X_test = test_df[features].values
        self.y_test = test_df["target"].values

        #self.volatility_train = train_df["VOLATILITY_90D"].values 

        return self.X_test

    def backtest(self, buy_probability_threshold:float=0.5, model_id:str='model') -> pd.DataFrame:
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

        Pr = run_inference_onnx(f'assets/{model_id}.onnx', self.X_test.astype(np.float32))

        self.portfolio["predicted_signal"] = (Pr > buy_probability_threshold).astype(int)
        print(self.portfolio.predicted_signal)
        self.portfolio["p_buy"] = Pr
        self.portfolio["p_sell"] = 1 - Pr

        # Initialize self.portfolio columns
        self.portfolio["cash_on_hand"] = self.initial_investment
        self.portfolio["share_value"] = 0
        self.portfolio["total_portfolio_value"] = self.initial_investment
        self.portfolio["cumulative_shares"] = 0
        self.portfolio["PnL"] = 0.0
        self.portfolio["position"] = "no position"

        # Initialize the first row based on initial investment
        # self.portfolio.iloc[0] = update_portfolio(self.portfolio.iloc[0], 
        #                                           self.portfolio.iloc[0], 
        #                                           self.initial_investment, 
        #                                           self.share_volume)

        # Iterate through the DataFrame to update each row
        for i in range(1, len(self.portfolio)):
            self.portfolio.iloc[i] = update_portfolio(self.portfolio.iloc[i], 
                                                      self.portfolio.iloc[i - 1], 
                                                      self.initial_investment, 
                                                      self.share_volume)

        # Return the relevant columns
        return self.portfolio[['DATE', 'CLOSE', 'RETURNS','cash_on_hand', 'share_value', 
                               "total_portfolio_value", "position", "cumulative_shares", 
                               "PnL", "predicted_signal"]]
        
