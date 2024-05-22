import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class RFDataset:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        stock_ids: list[str],
        start_date: str,
        end_date: str,
        features: list[str],
        horizon: int = 4,
        tau: float = 0.5,
    ):
        self.data = dataframe
        self.stock_ids = stock_ids
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.features = features
        self.horizon = horizon
        self.tau = tau
        self.processed_data = []
        self.preprocess()

    def preprocess(self):
        for stock_id in self.stock_ids:
            stock_data = self.data[(self.data["ID"] == stock_id)]

            # Calculate future returns
            stock_data["future_returns"] = stock_data["RETURNS"].diff(periods=self.horizon)

            # Calculate rolling indicators for momentum and volatility
            stock_data["momentum"] = stock_data["RETURNS"].rolling(window=self.horizon).mean()

            # Define the threshold based on volatility
            threshold = self.tau * stock_data["VOLATILITY_90D"]

            stock_data.dropna(inplace=True)

            threshold = threshold.reindex(stock_data.index)
            stock_data["future_returns"] = stock_data["future_returns"].reindex(stock_data.index)

            # Calculate labels based on future returns and other indicators
            conditions = [
                (stock_data["future_returns"] > threshold)
                & (stock_data["momentum"] > 0)
                & (stock_data["CLOSE_SMA_9D"] > stock_data["CLOSE_SMA_21D"]),
                (stock_data["future_returns"] < -threshold)
                & (stock_data["momentum"] < 0)
                & (stock_data["CLOSE_SMA_9D"] < stock_data["CLOSE_SMA_21D"]),
            ]
            choices = [1, -1]
            stock_data["label"] = np.select(conditions, choices, default=0)
            stock_data["label"] = stock_data["label"].map({-1: 0, 0: 1, 1: 2})

            self.processed_data.append(stock_data)

        self.processed_data = pd.concat(self.processed_data).reset_index(drop=True)

    def train_test_split_and_scale(self, test_start_date: str):
        test_start_date = pd.to_datetime(test_start_date)
        train_data = self.processed_data[self.processed_data["DATE"] < test_start_date]
        test_data = self.processed_data[self.processed_data["DATE"] >= test_start_date]

        X_train = train_data[self.features]
        y_train = train_data["label"]
        X_test = test_data[self.features]
        y_test = test_data["label"]

        scaler = StandardScaler().set_output(transform="pandas")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train, y_train = self.downsample_random(X_train, y_train)

        return X_train, X_test, y_train, y_test

    def downsample_random(self, X, y):
        df = pd.concat([pd.DataFrame(X, columns=self.features), y.reset_index(drop=True)], axis=1)
        min_count = df["label"].value_counts().min()
        df_downsampled = df.groupby("label").apply(lambda x: x.sample(min_count)).reset_index(drop=True)
        X_downsampled = df_downsampled[self.features]
        y_downsampled = df_downsampled["label"]
        return X_downsampled, y_downsampled
