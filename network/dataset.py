import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class StockDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        sequence_length: int = 90,
    ):
        self.df = df
        self.features = features[1:]
        self.target = target
        self.sequence_length = sequence_length
        self.grouped_data = self.prepare_data()

    def prepare_data(self):
        # Group data by ID and prepare sequences
        grouped = self.df.groupby("ID")
        sequences = []
        for _, group in grouped:
            # Ensure the group is sorted by date
            group = group.sort_values("DATE")
            # Get the last 'sequence_length' rows, or pad if necessary
            if len(group) < self.sequence_length:
                # Padding: Create a DataFrame with missing rows filled with zeros (or another padding value)
                pad_length = self.sequence_length - len(group)
                padded_data = pd.DataFrame(
                    np.zeros((pad_length, len(self.features) + 1)),
                    columns=self.features + [self.target],
                )
                group = pd.concat([padded_data, group], ignore_index=True)
            elif len(group) > self.sequence_length:
                group = group.iloc[
                    -self.sequence_length :
                ]  # Truncate to the last 'sequence_length' rows

            features_tensor = torch.tensor(
                group[self.features].values, dtype=torch.float
            )
            target_tensor = torch.tensor(group[self.target].values, dtype=torch.float)
            sequences.append((features_tensor, target_tensor))
        return sequences

    def __len__(self):
        return len(self.grouped_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.grouped_data[idx]
