import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Import your custom classes
from optimize import CustomLoss
from dataset import StockDataset


class StockModel:
    def __init__(self, filename: str):
        self.df = pd.read_csv(filename)
        self.df["DATE"] = pd.to_datetime(self.df["DATE"])
        self.df.sort_values("DATE", inplace=True)
        self.model = None
        self.scaler = StandardScaler().set_output(transform="pandas")
        self.dataset = None

    def preprocess_data(self):
        self.df["target"] = (self.df["RETURNS"] > 0).astype(int)

        features = ["ID", "CLOSE", "RETURNS", "VOLATILITY_90D"]
        self.df["VOLATILITY_90D"] = (
            self.df["VOLATILITY_90D"] / self.df["VOLATILITY_90D"].max()
        )

        # Define date split
        split_date = pd.Timestamp("2020-01-01")
        gap = pd.Timedelta(weeks=2)
        train_df = self.df[self.df["DATE"] < split_date]
        test_df = self.df[self.df["DATE"] >= split_date + gap]

        # Scale features
        self.scaler.fit(train_df[features[1:]])

        train_features = train_df
        train_features[features[1:]] = self.scaler.transform(train_df[features[1:]])

        test_features = test_df
        test_features[features[1:]] = self.scaler.transform(test_df[features[1:]])

        # Create dataset instances
        self.train_dataset = StockDataset(train_features, features, "target")
        self.test_dataset = StockDataset(test_features, features, "target")

    def train(self, epochs: int = 40, batch_size: int = 32):
        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        criterion = CustomLoss()

        for epoch in range(epochs):
            self.model.train()
            for data, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.model(
                    data[:, :3]
                )  # Use only non-volatility features for prediction
                loss = criterion(
                    outputs, targets, data[:, 2]
                )  # Volatility is the third feature
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def backtest(self):
        self.portfolio = (
            self.df[["DATE", "CLOSE", "RETURNS"]]
            .loc[self.df.DATE >= "2020-01-15"]
            .copy()
        )
        self.portfolio.reset_index(
            drop=True, inplace=True
        )  # Reset index to avoid any potential index issues

        # Evaluate the model and predict
        test_data_tensor = torch.tensor(self.X_test.values, dtype=torch.float)
        self.model.eval()
        with torch.no_grad():
            predicted_probabilities = self.model(test_data_tensor).squeeze()

        self.portfolio["predicted_signal"] = (predicted_probabilities > 0.5).numpy()
        self.portfolio["p"] = predicted_probabilities.numpy()
        self.portfolio["action"] = self.portfolio["predicted_signal"].diff()

        # Initialize investment and tracking variables
        initial_investment = 10000
        self.portfolio["portfolio_value"] = initial_investment
        in_position = False

        # Iterate through the DataFrame using iterrows (consider using vectorized operations for production optimization)
        for i, row in self.portfolio.iterrows():
            if row["action"] == 1:  # Buy signal
                in_position = True
            elif row["action"] == -1 and in_position:  # Sell signal
                in_position = False

            # Update portfolio value if in position
            if in_position:
                if i > 0:  # Ensure there is a previous value to reference
                    current_value = self.portfolio.at[i - 1, "portfolio_value"]
                    self.portfolio.at[i, "portfolio_value"] = current_value * (
                        1 + row["RETURNS"]
                    )

        self.portfolio["action_label"] = (
            self.portfolio["action"].map({1: "buy", -1: "sell"}).fillna("hold")
        )

        return self.portfolio[
            [
                "DATE",
                "CLOSE",
                "RETURNS",
                "predicted_signal",
                "p",
                "action",
                "action_label",
                "portfolio_value",
            ]
        ]


if __name__ == "__main__":
    stock_model = StockModel("../assets/data.csv")
    stock_model.preprocess_data()
    stock_model.train()
    action_df = stock_model.backtest()
    print(action_df)
    print(action_df.portfolio_value.max())
    print(action_df.portfolio_value.min())
    print(action_df.portfolio_value.mean())

    action_df.to_csv("res.csv")
