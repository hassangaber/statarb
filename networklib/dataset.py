import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        stock_ids: list[str],
        start_date: str,
        end_date: str,
        features: list[str],
        # target:str,
        horizon: int = 8,
        tau: float = 1.0,
    ):

        self.data = dataframe
        self.stock_ids = stock_ids
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.features = features
        # self.target = target
        self.horizon = horizon
        self.tau = tau

        self.scalers = {stock_id: StandardScaler().set_output(transform="pandas") for stock_id in self.stock_ids}
        self.processed_data = []
        self.preprocess()

    def preprocess(self):
        for stock_id in self.stock_ids:

            stock_data = self.data[
                (self.data["ID"] == stock_id) & (self.data["DATE"] >= self.start_date) & (self.data["DATE"] <= self.end_date)
            ].sort_values("DATE")

            stock_data.dropna(inplace=True)

            # Calculate future returns (log returns over the horizon)
            stock_data["future_returns"] = stock_data["RETURNS"].shift(-self.horizon)

            # Calculate rolling volatility based on log returns
            realized_variance = stock_data["RETURNS"].rolling(window=self.horizon).apply(lambda x: np.sum(x**2), raw=True)
            stock_data["realized_volatility"] = np.sqrt(realized_variance)

            # Define the threshold based on volatility
            threshold = self.tau * stock_data["realized_volatility"]
            stock_data.dropna(inplace=True)

            # Calculate labels based on future returns and volatility threshold
            conditions = [(stock_data["future_returns"] > threshold), (stock_data["future_returns"] < -threshold)]
            choices = [1, -1]
            stock_data["label"] = np.select(conditions, choices, default=0)

            self.scalers[stock_id].fit(stock_data[self.features])
            stock_data[self.features] = self.scalers[stock_id].transform(stock_data[self.features])

            self.processed_data.append(stock_data)

        self.processed_data = pd.concat(self.processed_data).reset_index(drop=True)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        row = self.processed_data.iloc[idx][self.features + ["label"]].astype(np.float64)
        features = torch.tensor(row[self.features].values, dtype=torch.float)
        label = torch.tensor(row["label"], dtype=torch.long)

        return features, label
