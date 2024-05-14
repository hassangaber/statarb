import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, stock_ids, start_date, end_date, features: list, target):
        self.data = dataframe
        self.stock_ids = stock_ids
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.features = features
        self.target = target
        self.scalers = {stock_id: StandardScaler().set_output(transform='pandas') for stock_id in self.stock_ids}
        self.processed_data = []
        self.preprocess()

    def preprocess(self):
        for stock_id in self.stock_ids:
            stock_data = self.data[(self.data['ID'] == stock_id) & 
                                   (self.data['DATE'] >= self.start_date) & 
                                   (self.data['DATE'] <= self.end_date)].sort_values('DATE')
            stock_data.dropna(inplace=True)
            self.scalers[stock_id].fit(stock_data[self.features])
            tempt = (stock_data[self.target] > 0).astype(int)
            stock_data = self.scalers[stock_id].transform(stock_data[self.features])
            stock_data['target'] = tempt
            self.processed_data.append(stock_data)

        self.processed_data = pd.concat(self.processed_data).reset_index(drop=True)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        row = self.processed_data.iloc[idx][self.features+['target']].astype(np.float64)
        features = torch.tensor(row[self.features].values, dtype=torch.float)
        target = torch.tensor(row['target'], dtype=torch.float)
        volatility = torch.tensor(row['VOLATILITY_90D'], dtype=torch.float)

        return features, target, volatility


