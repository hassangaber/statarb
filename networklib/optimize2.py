import torch
import torch.nn as nn

class ExcessReturnLoss(nn.Module):
    def __init__(self, risk_free_rate=0.0):
        super(ExcessReturnLoss, self).__init__()
        self.risk_free_rate = risk_free_rate

    def forward(self, returns, signals):
        """
        :param returns: Tensor of shape (batch_size, ), actual returns of the assets
        :param signals: Tensor of shape (batch_size, ), predicted signals (1 for buy, -1 for sell, 0 for hold)
        :return: Loss value
        """
        # Ensure returns and signals are broadcastable and compatible
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)
        if signals.dim() == 1:
            signals = signals.unsqueeze(1)
        
        # Check if dimensions match, otherwise raise an error
        if returns.size() != signals.size():
            raise ValueError(f"Shape mismatch: returns shape {returns.shape}, signals shape {signals.shape}")

        # Calculate excess returns based on the signals
        excess_returns = signals * (returns - self.risk_free_rate)

        # Calculate the mean and standard deviation of excess returns
        mean_excess_return = torch.mean(excess_returns)
        std_excess_return = torch.std(excess_returns)

        # Sharpe Ratio: mean excess return / standard deviation of excess return
        sharpe_ratio = mean_excess_return / (std_excess_return + 1e-8)

        # Loss is the negative Sharpe Ratio (we want to maximize Sharpe Ratio, so we minimize its negative)
        loss = -sharpe_ratio

        return loss
