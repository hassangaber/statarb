import torch
import torch.nn as nn
import torch.nn.functional as F

class VolatilityWeightedLoss(nn.Module):
    def __init__(self):
        super(VolatilityWeightedLoss, self).__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, volatility: torch.Tensor) -> torch.Tensor:
        predictions = predictions.squeeze()
        targets = targets.float()

        # Binary Cross Entropy loss for classification with dynamic weighting
        weight = 1 + volatility
        base_loss = F.binary_cross_entropy(predictions, targets)

        # Penalize large deviations, scaled by volatility
        #deviation_penalty = ((predictions - targets).abs() * volatility).mean()

        # Total loss combining base loss and deviation penalty
        #total_loss = base_loss + deviation_penalty

        return base_loss

