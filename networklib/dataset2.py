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
        horizon: int = 8,
        tau: float = 1.0,
        testing: bool = False,
    ):

        self.data = dataframe
        self.stock_ids = stock_ids
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.features = features
        self.horizon = horizon
        self.tau = tau
        self.testing = testing

        self.scalers = {stock_id: StandardScaler().set_output(transform="pandas") for stock_id in self.stock_ids}
        self.processed_data = []
        self.preprocess()

    def set_scaler(self, s: dict[str, StandardScaler]) -> None:
        self.scalers = s

    def get_scalers(self) -> dict[str, StandardScaler]:
        return self.scalers

    def preprocess(self):
        for stock_id in self.stock_ids:
            stock_data = self.data[
                (self.data["ID"] == stock_id) & (self.data["DATE"] >= self.start_date) & (self.data["DATE"] <= self.end_date)
            ].sort_values("DATE")

            stock_data.dropna(inplace=True)

            if not self.testing:
                self.scalers[stock_id].fit(stock_data[self.features])

            stock_data[self.features] = self.scalers[stock_id].transform(stock_data[self.features])

            # Calculate future returns (log returns over the horizon)
            stock_data["delta_RETURNS"] = stock_data["RETURNS"].diff(periods=self.horizon)

            # Calculate rolling indicators for momentum and volatility
            stock_data["momentum"] = stock_data["RETURNS"].rolling(window=self.horizon).mean()
            # realized_variance = stock_data['RETURNS'].rolling(window=self.horizon).var()
            # stock_data['volatility'] = np.sqrt(realized_variance)

            # Define the threshold based on volatility
            threshold = self.tau * stock_data["VOLATILITY_90D"]

            stock_data.dropna(inplace=True)

            # threshold = threshold.reindex(stock_data.index)
            # stock_data["future_returns"] = stock_data["future_returns"].reindex(stock_data.index)

            # Calculate labels based on future returns and other indicators
            conditions = [
                (stock_data["delta_RETURNS"] > threshold)(stock_data["momentum"] > 0)
                & (stock_data["CLOSE_SMA_3D"] > stock_data["CLOSE_SMA_9D"])
                & (stock_data["CLOSE_SMA_9D"] > stock_data["CLOSE_SMA_21D"]),
                (stock_data["delta_RETURNS"] < -threshold)(stock_data["momentum"] < 0)
                & (stock_data["CLOSE_SMA_3D"] < stock_data["CLOSE_SMA_9D"])
                & (stock_data["CLOSE_SMA_9D"] < stock_data["CLOSE_SMA_21D"]),
            ]
            choices = [2, 0]
            stock_data["label"] = np.select(conditions, choices, default=1)

            self.processed_data.append(stock_data)

        self.processed_data = pd.concat(self.processed_data).reset_index(drop=True)
        print(self.processed_data.label.value_counts() / self.processed_data.label.value_counts().sum())
        temp_X = self.processed_data.drop(columns=["label"])
        self.processed_data = self.downsample_random(temp_X, self.processed_data["label"], [0.35, 0.2, 0.45])

        print(self.processed_data.label.value_counts() / self.processed_data.label.value_counts().sum())

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        row = self.processed_data.iloc[idx][self.features + ["label"]].astype(np.float64)
        features = torch.tensor(row[self.features].values, dtype=torch.float)
        label = torch.tensor(row["label"], dtype=torch.long)

        return features, label

    def downsample_random(self, X, y, proportions):
        """
        Downsamples the signals according to the provided proportions.

        :param X: DataFrame of features
        :param y: Series of labels (0 for sell, 1 for hold, 2 for buy)
        :param proportions: List of proportions for sell, hold, and buy signals respectively
        :return: Downsampled DataFrame of features and Series of labels
        """
        assert sum(proportions) == 1, "Proportions must sum to 1"

        # Combine features and labels into a single DataFrame
        df = pd.concat([pd.DataFrame(X, columns=self.features), y.reset_index(drop=True)], axis=1)
        df.columns = list(df.columns[:-1]) + ["label"]

        # Calculate the target count for each label based on the total number of samples
        total_count = len(df)
        target_counts = {label: int(total_count * proportions[i]) for i, label in enumerate([0, 1, 2])}

        downsampled_df_list = []

        for label, target_count in target_counts.items():
            label_df = df[df["label"] == label]
            if len(label_df) > target_count:
                downsampled_df_list.append(label_df.sample(target_count, random_state=42))
            else:
                downsampled_df_list.append(label_df)

        # Concatenate all downsampled DataFrames
        df_downsampled = pd.concat(downsampled_df_list).reset_index(drop=True)
        df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the DataFrame

        X_downsampled = df_downsampled[self.features]
        y_downsampled = df_downsampled["label"]

        df = pd.concat([X_downsampled, y_downsampled], axis=1)

        return df
