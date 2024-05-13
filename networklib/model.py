import torch
import torch.nn.functional as F
import torch.nn as nn


class StockModel(nn.Module):

    def __init__(
        self, in_features: int = 4, hidden_1: int = 64, hidden_2: int = 32, out: int = 1
    ):

        super(StockModel, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)