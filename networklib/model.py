import torch
import torch.nn.functional as F
import torch.nn as nn


class StockModel(nn.Module):

    def __init__(
        self, in_features: int = 6, hidden_1: int = 128, hidden_2: int = 64, out: int = 1
    ):

        super(StockModel, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, out)

        self.init_weights()

    def init_weights(self):
       # nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)