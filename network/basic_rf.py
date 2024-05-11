import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

pd.options.display.float_format = '{:,.2f}'.format

class VolatilityWeightedLoss(nn.Module):
    def __init__(self):
        super(VolatilityWeightedLoss, self).__init__()

    def forward(self, predictions, targets, volatility):
        # Calculate cross-entropy loss for a multi-class problem
        base_loss = F.cross_entropy(predictions, targets)
        # Penalize the loss based on the mean volatility
        volatility_penalty = volatility.mean() * base_loss
        return base_loss + volatility_penalty

class StockModel(nn.Module):
    def __init__(self):
        super(StockModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # Three outputs for the classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)  # Apply softmax to convert to probabilities

class PortfolioPrediction:
    def __init__(self, filename, stock_id, train_end_date, test_start_date, start_date):
        self.df = pd.read_csv(filename)
        self.df['DATE'] = pd.to_datetime(self.df['DATE'])

        self.df.sort_values('DATE', inplace=True)
        self.stock_id = stock_id
        self.train_end_date = pd.to_datetime(train_end_date)
        self.test_start_date = pd.to_datetime(test_start_date)
        self.start_date = pd.to_datetime(start_date)
        self.model = None
        self.portfolio = pd.DataFrame()
        self.scaler = StandardScaler().set_output(transform='pandas')

    def preprocess_data(self):
        self.df = self.df[(self.df.ID == self.stock_id) & (self.df.DATE >= self.start_date)]
        self.df.dropna(inplace=True)
        # Encoding: 0 = hold, 1 = buy, 2 = sell
        self.df['target'] = self.df['RETURNS'].apply(lambda x: 1 if x > 0 else 2 if x < 0 else 0)
        features = ['CLOSE', 'RETURNS', 'VOLATILITY_90D']
        
        train_df = self.df[self.df['DATE'] <= self.train_end_date]
        test_df = self.df[self.df['DATE'] >= self.test_start_date]

        self.scaler.fit(train_df[features])
        train_df[features] = self.scaler.transform(train_df[features])
        test_df[features] = self.scaler.transform(test_df[features])

        self.X_train = train_df[features].values
        self.y_train = train_df['target'].values
        self.X_test = test_df[features].values
        self.y_test = test_df['target'].values
        self.volatility_train = train_df['VOLATILITY_90D'].values  # For loss calculation

    def train(self, epochs=500, batch_size=32):
        self.model = StockModel()
        criterion = VolatilityWeightedLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        train_data = TensorDataset(torch.tensor(self.X_train, dtype=torch.float),
                                   torch.tensor(self.y_train, dtype=torch.long),
                                   torch.tensor(self.volatility_train, dtype=torch.float))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()
            for data, targets, volatility in train_loader:
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets, volatility)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    def backtest(self, initial_investment=10000, share_volume=5):
        self.portfolio = self.df[['DATE', 'CLOSE', 'RETURNS']].loc[self.df['DATE'] >= self.test_start_date].copy()
        self.portfolio.sort_values(by='DATE', inplace=True)
        self.portfolio.reset_index(drop=True, inplace=True)
        test_data_tensor = torch.tensor(self.X_test, dtype=torch.float)
        self.model.eval()
        with torch.no_grad():
            predicted_probabilities = self.model(test_data_tensor)
        
        self.portfolio['predicted_signal'] = predicted_probabilities.max(1)[1]  # Max log-probability indices
        self.portfolio['p_buy'] = predicted_probabilities[:, 1]  # Probability of buy
        self.portfolio['p_sell'] = predicted_probabilities[:, 2]  # Probability of sell
        self.portfolio['p_hold'] = predicted_probabilities[:, 0]  # Probability of hold
        
        self.portfolio['portfolio_value'] = initial_investment
        self.portfolio['cumulative_shares'] = 0  # Initialize cumulative shares column
        self.portfolio['cumulative_share_cost'] = 0  # Initialize cumulative share cost column
        number_of_shares = 0
        cumulative_share_cost = 0  # Total cost of shares held

        for i, row in self.portfolio.iterrows():
            if i == 0:
                continue

            if row['predicted_signal'] == 1 and self.portfolio.at[i - 1, 'portfolio_value'] > row['CLOSE'] * share_volume:
                # Buy shares
                number_of_shares += share_volume
                purchase_cost = row['CLOSE'] * share_volume
                self.portfolio.at[i, 'portfolio_value'] -= purchase_cost
                cumulative_share_cost += purchase_cost

            elif row['predicted_signal'] == 2 and number_of_shares >= share_volume:
                # Sell shares
                number_of_shares -= share_volume
                sale_proceeds = row['CLOSE'] * share_volume
                self.portfolio.at[i, 'portfolio_value'] += sale_proceeds
                cumulative_share_cost -= row['CLOSE'] * share_volume

            # Update cumulative share and cost tracking
            self.portfolio.at[i, 'cumulative_shares'] = number_of_shares
            self.portfolio.at[i, 'cumulative_share_cost'] = cumulative_share_cost

            # Update portfolio value for current market value of shares
            if number_of_shares > 0:
                current_market_value = number_of_shares * row['CLOSE']
                self.portfolio.at[i, 'portfolio_value'] += current_market_value - cumulative_share_cost

        return self.portfolio[['DATE', 'CLOSE', 'RETURNS', 'p_buy', 'p_sell', 'p_hold', 'predicted_signal', 'cumulative_shares', 'cumulative_share_cost', 'portfolio_value']]






# import pandas as pd
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, TensorDataset
# import torch.nn as nn
# from sklearn.preprocessing import StandardScaler
# import torch.nn.functional as F

# class VolatilityWeightedLoss(nn.Module):
#     def __init__(self):
#         super(VolatilityWeightedLoss, self).__init__()

#     def forward(self, predictions, targets, volatility):
#         base_loss = F.cross_entropy(predictions, targets)  # Use cross-entropy for multi-class
#         volatility_penalty = volatility.mean() * base_loss  # Weight loss by average volatility
#         return base_loss + volatility_penalty



# class ModifiedModel(nn.Module):
#     def __init__(self):
#         super(ModifiedModel, self).__init__()
#         self.fc1 = nn.Linear(3, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 3)  # Three outputs for buy, hold, sell

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.softmax(x, dim=1)  # Softmax along the second dimension to get probabilities



# class StockModel:
#     def __init__(self, filename, stock_id, train_end_date, test_start_date, start_date='2015-01-01'):
#         self.df = pd.read_csv(filename)
#         self.df['DATE'] = pd.to_datetime(self.df['DATE'])
#         self.stock_id = stock_id
#         self.train_end_date = pd.to_datetime(train_end_date)
#         self.test_start_date = pd.to_datetime(test_start_date)
#         self.start_date = pd.to_datetime(start_date)
#         self.model = None
#         self.portfolio = pd.DataFrame()
#         self.scaler = StandardScaler().set_output(transform='pandas')

#     def preprocess_data(self):
#         # Filter data by stock ID and date range
#         self.df = self.df[(self.df.ID == self.stock_id) & (self.df.DATE >= self.start_date)]
#         self.df.dropna(inplace=True)

#         #self.df['target'] = np.where(self.df['RETURNS'] > 0, 1, -1) 
#         # Adjust targets from -1 and 1 to 0 and 1
#         self.df['target'] = (self.df['RETURNS'] > 0).astype(int)  # Will set 0 for False, 1 for True


#         features = ['CLOSE', 'RETURNS', 'VOLATILITY_90D']
#         self.df = self.df[features + ['DATE', 'target']]

#         # Split data
#         train_df = self.df[self.df['DATE'] <= self.train_end_date]
#         test_df = self.df[self.df['DATE'] >= self.test_start_date]
        
#         self.X_train, self.y_train = train_df[features], train_df['target']
#         self.X_test, self.y_test = test_df[features], test_df['target']

#         # Scale features
#         self.scaler.fit(self.X_train)
#         self.X_train = self.scaler.transform(self.X_train)
#         self.X_test = self.scaler.transform(self.X_test)

#         print('y: ',self.y_test.values)
#         return (self.X_train, self.y_train, self.X_test, self.y_test)

#     def train(self, epochs=500, batch_size=32):
#         self.model = nn.Sequential(
#             nn.Linear(3, 64), nn.ReLU(),
#             nn.Linear(64, 32), nn.ReLU(),
#             nn.Linear(32, 1), nn.Sigmoid()
#         )
#         criterion = nn.BCELoss()
#         #criterion = HingeLoss()
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
#         train_data = TensorDataset(torch.tensor(self.X_train.values, dtype=torch.float), torch.tensor(self.y_train.values, dtype=torch.float))
#         train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

#         for epoch in range(epochs):
#             self.model.train()
#             for data, targets in train_loader:
#                 optimizer.zero_grad()
#                 outputs = self.model(data).squeeze()
#                 loss = criterion(outputs, targets)
#                 loss.backward()
#                 optimizer.step()
#             #print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

#     def backtest(self):
#         # Ensure DataFrame for the portfolio is properly initialized
#         self.portfolio = self.df[['DATE', 'CLOSE', 'RETURNS']].loc[self.df.DATE >= self.test_start_date].copy()
#         self.portfolio.reset_index(drop=True, inplace=True)  # Reset index to avoid any potential index issues

#         # Evaluate the model and predict
#         test_data_tensor = torch.tensor(self.X_test.values, dtype=torch.float)
#         self.model.eval()
#         with torch.no_grad():
#             predicted_probabilities = self.model(test_data_tensor).squeeze()
        
#         self.portfolio['predicted_signal'] = (predicted_probabilities > 0.5).numpy()
#         self.portfolio['p_buy'] = predicted_probabilities.numpy()
#         self.portfolio['p_sell'] = 1 - predicted_probabilities.numpy()
#         self.portfolio['action'] = self.portfolio['predicted_signal'].diff()[1:].astype(int)

#         print('action, ',self.portfolio['action'])

#         # Initialize investment and tracking variables
#         initial_investment = 10000
#         self.portfolio['portfolio_value'] = initial_investment
#         in_position = False

#         # Iterate through the DataFrame using iterrows (consider using vectorized operations for production optimization)
#         for i, row in self.portfolio.iterrows():
#             if row['action'] == 1:  # Buy signal
#                 in_position = True
#             elif row['action'] == -1 and in_position:  
#                 in_position = False

#             # Update portfolio value if in position
#             if in_position:
#                 if i > 0:  # Ensure there is a previous value to reference
#                     current_value = self.portfolio.at[i - 1, 'portfolio_value']
#                     self.portfolio.at[i, 'portfolio_value'] = current_value * (1 + row['RETURNS'])
        
#         self.portfolio['action_label'] = self.portfolio['action'].apply(lambda x: 'buy' if x > 0 else ('sell' if x < 0 else 'hold'))

#         return self.portfolio[['DATE', 'CLOSE', 'RETURNS', 'predicted_signal', 'p_sell', 'p_buy', 'action', 'action_label','portfolio_value']]



# # # Example usage
# # stock_model = StockModel('../assets/data.csv')
# # stock_model.preprocess_data()
# # stock_model.train()
# # action_df = stock_model.backtest()
# # print(action_df)
# # print(action_df.portfolio_value.max())
# # print(action_df.portfolio_value.min())
# # print(action_df.portfolio_value.mean())

# # action_df.to_csv('res.csv')