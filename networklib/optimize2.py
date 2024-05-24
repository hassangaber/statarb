import torch
import torch.nn as nn


class ExcessReturnLoss(nn.Module):
    def __init__(self, risk_free_rate=0.0):
        super(ExcessReturnLoss, self).__init__()
        self.risk_free_rate = risk_free_rate

    def forward(self, returns, probabilities):
        """
        :param returns: Tensor of shape (batch_size, ), actual returns of the assets
        :param probabilities: Tensor of shape (batch_size, 3), predicted probabilities for each class (buy, sell, hold)
        :return: Loss value
        """
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)

        # Extract probabilities
        sell_prob, hold_prob, buy_prob = probabilities[:, 0], probabilities[:, 1], probabilities[:, 2]

        # Calculate signals as buy_prob - sell_prob
        signals = buy_prob - sell_prob

        # Calculate excess returns based on the signals
        excess_returns = signals * (returns - self.risk_free_rate)

        # Calculate the mean and standard deviation of excess returns
        mean_excess_return = torch.mean(excess_returns)
        std_excess_return = torch.std(excess_returns)

        # Sharpe Ratio: mean excess return / standard deviation of excess return
        sharpe_ratio = mean_excess_return / (std_excess_return + 1e-8)

        # Loss is the negative Sharpe Ratio (we want to maximize Sharpe Ratio, so we minimize its negative)
        loss = sharpe_ratio

        return loss


import torch
import torch.nn as nn


class CustomTradingLoss(nn.Module):
    def __init__(self, risk_free_rate=0.0, alpha=0.5):
        super(CustomTradingLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
        self.alpha = alpha  # Weighting factor for combining cross-entropy and return-based loss

    def forward(self, probabilities, true_labels):
        """
        :param returns: Tensor of shape (batch_size, ), actual returns of the assets
        :param probabilities: Tensor of shape (batch_size, 3), predicted probabilities for each class (buy, sell, hold)
        :param true_labels: Tensor of shape (batch_size, ), true labels (0 for sell, 1 for hold, 2 for buy)
        :return: Loss value
        """

        # Extract probabilities
        sell_prob, hold_prob, buy_prob = probabilities[:, 0], probabilities[:, 1], probabilities[:, 2]

        cross_entropy_loss = nn.CrossEntropyLoss()(probabilities, true_labels)

        # Combine the cross-entropy loss with the Sharpe ratio (or any other return-based metric)
        # combined_loss = self.alpha * cross_entropy_loss - (1 - self.alpha) * sharpe_ratio

        return cross_entropy_loss
