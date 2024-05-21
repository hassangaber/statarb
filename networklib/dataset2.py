import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 stock_ids: list[str], 
                 start_date: str, 
                 end_date: str, 
                 features: list[str], 
                 horizon: int = 8, 
                 tau: float = 1.0,
                 testing:bool = False):
        
        self.data = dataframe
        self.stock_ids = stock_ids
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.features = features
        self.horizon = horizon
        self.tau = tau
        self.testing = testing

        self.scalers = {stock_id: StandardScaler().set_output(transform='pandas') for stock_id in self.stock_ids}
        self.processed_data = []
        self.preprocess()

    def set_scaler(self, s: dict[str, StandardScaler]) -> None:
        self.scalers = s

    def get_scalers(self) -> dict[str, StandardScaler]:
        return self.scalers

    def preprocess(self):
        for stock_id in self.stock_ids:
            stock_data = self.data[(self.data['ID'] == stock_id) & 
                                   (self.data['DATE'] >= self.start_date) & 
                                   (self.data['DATE'] <= self.end_date)].sort_values('DATE')
            
            stock_data.dropna(inplace=True)

            if not self.testing: 
                self.scalers[stock_id].fit(stock_data[self.features])
                
            stock_data[self.features] = self.scalers[stock_id].transform(stock_data[self.features])

            # Calculate future returns (log returns over the horizon)
            stock_data['future_returns'] = stock_data['RETURNS'].diff(periods=self.horizon)
            
            # Calculate rolling indicators for momentum and volatility
            stock_data['momentum'] = stock_data['RETURNS'].rolling(window=self.horizon).mean()
            # realized_variance = stock_data['RETURNS'].rolling(window=self.horizon).var()
            # stock_data['volatility'] = np.sqrt(realized_variance)
            
            # Define the threshold based on volatility
            threshold = self.tau * stock_data['VOLATILITY_90D']
            
            stock_data.dropna(inplace=True)

            threshold = threshold.reindex(stock_data.index)
            stock_data['future_returns'] = stock_data['future_returns'].reindex(stock_data.index)
            
            # Calculate labels based on future returns and other indicators
            conditions = [
                (stock_data['future_returns'] > threshold) & 
                (stock_data['momentum'] > 0) & 
                (stock_data["CLOSE_SMA_9D"] > stock_data["CLOSE_SMA_21D"]) 
               ,
                
                (stock_data['future_returns'] < -threshold) & 
                (stock_data['momentum'] < 0) & 
                (stock_data["CLOSE_SMA_9D"] < stock_data["CLOSE_SMA_21D"]) 
            ]
            choices = [1, -1]
            stock_data['label'] = np.select(conditions, choices, default=0)
        

            self.processed_data.append(stock_data)

        self.processed_data = pd.concat(self.processed_data).reset_index(drop=True)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        row = self.processed_data.iloc[idx][self.features+['label']].astype(np.float64)
        features = torch.tensor(row[self.features].values, dtype=torch.float)
        label = torch.tensor(row['label'], dtype=torch.long) 
        return features, label

