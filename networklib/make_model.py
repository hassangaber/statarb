import torch
import pandas as pd

from torch.utils.data import DataLoader
import torch.onnx

from optimize2 import ExcessReturnLoss, CustomTradingLoss
from dataset2 import TimeSeriesDataset
from model2 import TradingSignalNet
from model3 import TemporalTradingSignalNet


class MakeModel:
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
        self.features = [
            "CLOSE",
            "VOLUME",
            "HIGH",
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
            "RETURNS",
            "HIGH_ROC_360D",
            "VOLATILITY_90D_ROC_360D",
        ]
        end_date = self.train_end_date if is_train else self.test_start_date

        dataset = TimeSeriesDataset(self.df, self.stock_ids, self.start_date, end_date, self.features, horizon=5,tau=1e-6)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return data_loader

    def train(self) -> torch.nn.Module:
        self.train_loader = self.make_data_loader(is_train=True)
        self.model = TradingSignalNet(input_dim=len(self.features), hidden_dim=128, output_dim=3)
        criterion = CustomTradingLoss(risk_free_rate=0.041/252)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, maximize=False)
        self.model.train()
        for epoch in range(self.epochs):
            for data, targets in self.train_loader:
                data = data.float()
                targets = targets.float()

                optimizer.zero_grad()
                outputs = self.model(data) # .squeeze()
                loss = criterion(outputs, targets.long())
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.5f}")

        return self.model


features = [
    "CLOSE",
    "VOLUME",
    "HIGH",
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
