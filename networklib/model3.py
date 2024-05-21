import torch
import torch.nn as nn
import torch.optim as optim

class TemporalTradingSignalNet(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=1, kernel_size=3):
        super(TemporalTradingSignalNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=1)
        self.adaptivepool = nn.AdaptiveAvgPool1d(output_size=1)  # Adaptive pooling to fixed size
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  # Adjust input size after pooling
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Assuming input x is of shape (batch_size, sequence_length, num_features)
        if x.dim() == 2:  # Add sequence length dimension if missing
            x = x.unsqueeze(2)  # Shape becomes (batch_size, num_features, 1)
        elif x.dim() == 3:  # Permute to (batch_size, num_features, sequence_length)
            x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.adaptivepool(x)  # Apply adaptive pooling
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)  # Apply tanh to constrain output between -1 and 1
        return x

    def _initialize_weights(self):
        # Apply Xavier initialization to all linear and convolutional layers
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        # Initialize biases to zero
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

