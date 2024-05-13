import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from optimize import VolatilityWeightedLoss
from model import StockModel

pd.options.display.float_format = "{:,.2f}".format


class PortfolioPrediction:
    def __init__(
        self,
        filename: str,
        stock_id: str,
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
        self.df.sort_values("DATE", inplace=True)

        self.stock_id = stock_id
        self.train_end_date = pd.to_datetime(train_end_date)
        self.test_start_date = pd.to_datetime(test_start_date)
        self.start_date = pd.to_datetime(start_date)

        self.model = None
        self.train_loader = None
        self.portfolio = pd.DataFrame()
        self.scaler = StandardScaler().set_output(transform="pandas")

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.initial_investment = initial_investment
        self.share_volume = share_volume

    def preprocess_data(self) -> None:

        self.df = self.df[(self.df.ID == self.stock_id) & (self.df.DATE >= self.start_date)].sort_values('DATE')
        self.df.dropna(inplace=True)

        self.df["target"] = (self.df["RETURNS"] > 0).astype(int).values

        features = ["CLOSE", "VOLATILITY_90D", "VOLUME", "HIGH"]

        train_df = self.df[self.df["DATE"] <= self.train_end_date]
        #test_df = self.df[self.df["DATE"] >= self.test_start_date]

        self.scaler.fit(train_df[features])
        train_df[features] = self.scaler.transform(train_df[features])
        #test_df[features] = self.scaler.transform(test_df[features])

        self.X_train = train_df[features].values
        self.y_train = train_df["target"].values

        #self.X_test = test_df[features].values
        #self.y_test = test_df["target"].values

        self.volatility_train = train_df["VOLATILITY_90D"].values 

    def make_tensor_dataloader(self) -> DataLoader:

        train_data = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float),
            torch.tensor(self.y_train, dtype=torch.long),
            torch.tensor(self.volatility_train, dtype=torch.float),
        )
        train_data
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)

        return self.train_loader

    def train(self) -> torch.nn.Module:
        self.model = StockModel()
        criterion = VolatilityWeightedLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.make_tensor_dataloader()

        self.model.train()
        for epoch in range(self.epochs):
            for data, targets, volatility in self.train_loader:
                data = data.float()
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets, volatility)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")

        return self.model

    def backtest(self) -> pd.DataFrame:
        self.portfolio = (
            self.df[["DATE", "CLOSE", "RETURNS"]]
            .loc[self.df["DATE"] >= self.test_start_date]
            .copy()
        )
        self.portfolio.sort_values(by="DATE", inplace=True)
        self.portfolio.reset_index(drop=True, inplace=True)

        test_data_tensor = torch.tensor(self.X_test, dtype=torch.float)
        self.model.eval()
        with torch.no_grad():
            predicted_probabilities = self.model(test_data_tensor)

        self.portfolio["predicted_signal"] = (predicted_probabilities > 0.5).int() 
        self.portfolio["p_buy"] = predicted_probabilities  # Probability of buy
        self.portfolio["p_sell"] = 1-predicted_probabilities  # Probability of sell

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
                self.portfolio.at[i, "alpha"] = round(portfolio_daily_return,3) # - benchmark_daily_return

        return self.portfolio[["DATE",
                                "CLOSE",
                                "RETURNS",
                                "p_buy",
                                "p_sell",
                                "predicted_signal",
                                "cumulative_shares",
                                "cumulative_share_cost",
                                "portfolio_value",
                                "alpha",]]

import torch
import torch.onnx

def convert_pytorch_to_onnx(model, input_size=(64,4), onnx_file_path='../assets/model.onnx'):
    # Set the model to evaluation mode
    model.eval()

    # An example input you would normally provide to your model's forward() method.
    example_input = torch.randn(input_size)

    # Export the model
    torch.onnx.export(model,               # model being run
                      example_input,       # model input (or a tuple for multiple inputs)
                      onnx_file_path,      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})
    print(f"Model has been converted to ONNX and saved to {onnx_file_path}")


import onnxruntime as ort
import numpy as np

def run_inference_onnx(model_path, input_data):
    # Create an inference session
    sess = ort.InferenceSession(model_path)

    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)
    if len(input_data.shape) == 1:
        input_data = np.expand_dims(input_data, axis=0) 

    # Prepare the input to the model according to the expected model inputs.
    input_name = sess.get_inputs()[0].name
    #input_data = input_data.dtype(np.float32)  # Ensure correct data type

    # Run the model and get the output
    result = sess.run(None, {input_name: input_data})
    return result[0]


if __name__ == "__main__":
    P = PortfolioPrediction(
        filename='../assets/data.csv',
        stock_id='AAPL',
        train_end_date='2024-01-01',
        test_start_date='2024-02-1',
        start_date='2011-01-01',
        batch_size=64,
        epochs=600,
        lr=3e-4,
        weight_decay=0.0001
    )

    P.preprocess_data()
    T = P.make_tensor_dataloader()
    MODEL = P.train()
    convert_pytorch_to_onnx(MODEL)

    for data, targets, volatility in T:
        R = run_inference_onnx(model_path='../assets/model.onnx',input_data=data.float())
        break

    print(R)
    