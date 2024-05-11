import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

data = pd.read_csv("../assets/data.csv")
data.fillna(0, inplace=True)
data["DATE"] = pd.to_datetime(data["DATE"])
data = data[data["ID"] == "AAPL"]
data = data[data["DATE"] > "2017-01-01"].loc[data.DATE < "2023-01-01"]

features = data
# Calculate future returns for a horizon, e.g., 5 days
future_period = 5
data["future_return"] = data["CLOSE"].shift(-future_period) / data["CLOSE"] - 1
data.dropna(inplace=True)  # Drop last rows with no future return data

targets = data[["DATE", "future_return"]]

X_train, X_valid = (
    features[features.DATE < "2022-01-01"].values,
    features[features.DATE >= "2022-02-01"].values,
)
y_train, y_valid = (
    targets[targets.DATE < "2022-01-01"].values,
    targets[targets.DATE >= "2022-02-01"].values,
)
print(X_train[0, 3:])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train[:, 3:])
X_valid = scaler.fit_transform(X_valid[:, 3:])

print(type(y_train[:, 1][0]))

# Convert to PyTorch tensors
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float),
    torch.tensor(y_train[:, 1].astype(float), dtype=torch.float),
)
valid_dataset = TensorDataset(
    torch.tensor(X_valid, dtype=torch.float),
    torch.tensor(y_valid[:, 1].astype(float), dtype=torch.float),
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)


# Define the model
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.layer1 = nn.Linear(55, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output(x)


model = Regressor()
optimizer = optim.Adam(model.parameters(), lr=3e-4)


# Custom loss function: Profit-Loss Maximization
def custom_loss(
    outputs,
    targets,
    transaction_cost=0.001,
    risk_free_rate=0.041,
    annual_trading_days=252,
):
    profit = 100 * (targets - transaction_cost) * outputs
    profit_volatility = torch.std(profit)
    mean_profit = torch.mean(profit)
    sharpe_ratio = (
        (mean_profit - risk_free_rate)
        / profit_volatility
        * torch.sqrt(torch.tensor(annual_trading_days))
    )

    return -sharpe_ratio  # Minimize negative Sharpe Ratio


# Training function
def train(model, train_loader, valid_loader, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, returns in train_loader:
            optimizer.zero_grad()
            predictions = model(features).squeeze()
            loss = custom_loss(predictions, returns)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss / len(train_loader)}"
        )

        # Validation phase
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for features, returns in valid_loader:
                predictions = model(features).squeeze()
                loss = custom_loss(predictions, returns)
                valid_loss += loss.item()
            print(f"Validation Loss: {valid_loss / len(valid_loader)}")


# Run training and validation
if __name__ == "__main__":
    train(model, train_loader, valid_loader, optimizer)
