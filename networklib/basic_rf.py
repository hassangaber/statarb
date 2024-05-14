import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

import onnxruntime as ort
import torch.onnx

from optimize import VolatilityWeightedLoss
from model import StockModel
from dataset import TimeSeriesDataset

class PortfolioPrediction:
    def __init__(
        self,
        filename: str,
        stock_ids: list,
        train_end_date: str,
        test_start_date: str,
        start_date: str,
        batch_size: int = 16,
        epochs: int = 50,
        lr: int = 0.01,
        weight_decay: float = 0.0001,
        initial_investment: int = 10000,
        share_volume: int = 5,
    ):
        self.df = pd.read_csv(filename)
        self.df["DATE"] = pd.to_datetime(self.df["DATE"])
        self.stock_ids = stock_ids
        self.train_end_date = pd.to_datetime(train_end_date)
        self.test_start_date = pd.to_datetime(test_start_date)
        self.start_date = pd.to_datetime(start_date)

        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.portfolio = pd.DataFrame()

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.initial_investment = initial_investment
        self.share_volume = share_volume

    def make_data_loader(self, is_train=True):
        features = ["CLOSE", "VOLATILITY_90D", "VOLUME", "HIGH", "CLOSE_ROC_3D", "CLOSE_EWMA_120D"]
        target = "RETURNS"
        end_date = self.train_end_date if is_train else self.test_start_date
        
        dataset = TimeSeriesDataset(self.df, self.stock_ids, self.start_date, end_date, features, target)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return data_loader

    def train(self) -> torch.nn.Module:
        self.train_loader = self.make_data_loader(is_train=True)
        self.model = StockModel()
        criterion = VolatilityWeightedLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        v=0
        self.model.train()  
        for epoch in range(self.epochs):
            for data, targets, volatility in self.train_loader:
                data = data.float()
                targets = targets.float()

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets, volatility)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.5f}")

        return self.model

    def backtest(self) -> pd.DataFrame:
        self.test_loader = self.make_data_loader(is_train=False)
        self.portfolio = (
            self.df[["DATE", "CLOSE", "RETURNS"]]
            .loc[self.df["DATE"] >= self.test_start_date]
            .copy()
        )
        self.portfolio.sort_values(by="DATE", inplace=True)
        self.portfolio.reset_index(drop=True, inplace=True)

        all_predictions = []
        with torch.no_grad():
            for data, _, _ in self.test_loader:
                outputs = self.model(data.float())
                all_predictions.extend(outputs.squeeze().tolist())

        all_predictions = all_predictions[:len(self.portfolio)]
        self.portfolio["predicted_signal"] = (torch.tensor(all_predictions) > 0.5).int()
        print(self.portfolio.predicted_signal)
        self.portfolio["p_buy"] = all_predictions
        self.portfolio["p_sell"] = 1 - torch.tensor(all_predictions)

        self.portfolio["portfolio_value"] = self.initial_investment
        self.portfolio["cash_on_hand"] = self.initial_investment
        self.portfolio["cumulative_shares"] = 0
        self.portfolio["cumulative_share_cost"] = 0
        number_of_shares = 0
        cumulative_share_cost = 0

        for i, row in self.portfolio.iterrows():
            if i == 0:
                continue

            close_price = row["CLOSE"]

            if row["predicted_signal"] == 1 and self.portfolio.at[i - 1, "cash_on_hand"] >= close_price * self.share_volume:
                # Buy shares
                number_of_shares += self.share_volume
                purchase_cost = close_price * self.share_volume
                self.portfolio.at[i, "cash_on_hand"] -= purchase_cost
                cumulative_share_cost += purchase_cost
                #self.portfolio.at["cumulative_shares"] += number_of_shares
                #self.portfolio.at[i, 'portfolio_value'] = self.portfolio.at[i, "cash_on_hand"] + close_price * self.portfolio.at[i, "cumulative_shares"]

            elif row["predicted_signal"] == 0 and number_of_shares >= self.share_volume:
                # Sell shares
                number_of_shares -= self.share_volume
                sale_proceeds = close_price * self.share_volume
                self.portfolio.at[i, "cash_on_hand"] += sale_proceeds
                cumulative_share_cost -= close_price * self.share_volume
                #self.portfolio.at["cumulative_shares"] -= number_of_shares
                #self.portfolio.at[i, 'portfolio_value'] = self.portfolio.at[i, "cash_on_hand"] + close_price * self.portfolio.at[i, "cumulative_shares"]

            # Update cumulative share and cost tracking
            self.portfolio.at[i, "cumulative_shares"] = number_of_shares
            self.portfolio.at[i, "cumulative_share_cost"] = cumulative_share_cost

            # Calculate the current market value of shares
            current_market_value = number_of_shares * close_price

            # Update portfolio value: cash on hand + market value of shares
            self.portfolio.at[i, "portfolio_value"] = self.portfolio.at[i, "cash_on_hand"] + current_market_value

            # Calculate alpha as the difference between portfolio return and benchmark return
            if i > 0:
                portfolio_daily_return = (self.portfolio.at[i, "portfolio_value"] / self.portfolio.at[0, "portfolio_value"] - 1)
                self.portfolio.at[i, "alpha"] = round(portfolio_daily_return, 3)

            # Handle cases where selling is attempted but no shares are held
            if row["predicted_signal"] == 0 and number_of_shares == 0:
                print(f"No position to sell at index {i} on date {row['DATE']}")


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

def convert_pytorch_to_onnx(model, input_size=(64, 6), onnx_file_path='../assets/model.onnx'):
    model.eval()
    example_input = torch.randn(input_size)
    torch.onnx.export(model, example_input, onnx_file_path, export_params=True, opset_version=10, do_constant_folding=True,
                      input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"Model has been converted to ONNX and saved to {onnx_file_path}")

def run_inference_onnx(model_path, input_data):
    sess = ort.InferenceSession(model_path)
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)
    if len(input_data.shape) == 1:
        input_data = np.expand_dims(input_data, axis=0)
    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: input_data})
    return result[0]

if __name__ == "__main__":
    P = PortfolioPrediction(
        filename='../assets/data.csv',
        stock_ids=['AAPL','NVDA','COST','XOM','T','JPM','JNJ','GE'],
        train_end_date='2023-12-01',
        test_start_date='2024-02-01',
        start_date='2015-01-01',
        batch_size=128,
        epochs=5,
        lr=0.01,
        weight_decay=0
    )

    MODEL = P.train()
    convert_pytorch_to_onnx(MODEL)

    P.backtest()
