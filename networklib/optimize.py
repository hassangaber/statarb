import torch.nn as nn
import torch.nn.functional as F
import torch


class VolatilityWeightedLoss(nn.Module):
    def __init__(self):
        super(VolatilityWeightedLoss, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        penalty=0.3

        predictions = predictions.squeeze()
        targets=targets.float()
        base_loss = F.binary_cross_entropy(predictions, targets)
        volatility_penalty = volatility.mean() * base_loss

        return (1-penalty)*base_loss + penalty*volatility_penalty