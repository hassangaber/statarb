import torch.nn as nn
import torch

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.base_loss = nn.BCELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        base_loss = self.base_loss(predictions, targets)
        volatility_penalty = torch.mean(volatility * base_loss)
        return base_loss + volatility_penalty
