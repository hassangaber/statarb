import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data: pd.DataFrame, sequence_length: int):
        """
        Args:
            data (pd.DataFrame): The dataframe containing the features and labels.
            sequence_length (int): The length of each sequence.
        """
        self.sequence_length = sequence_length
        self.data = data.sort_values(['ID', 'DATE']).reset_index(drop=True)
        
        self.data_groups = self.data.groupby('ID')
        
        self.sequences = []
        self.labels = []
        self.prepare_sequences()
    
    def prepare_sequences(self):
        for _, group in self.data_groups:
            group_features = group.drop(columns=['TARGET', 'ID', 'DATE']).dropna().values
            group_labels = group['TARGET'].values

            
            num_sequences = (len(group) - self.sequence_length) // self.sequence_length
            for i in range(num_sequences):
                start_idx = i * self.sequence_length
                end_idx = start_idx + self.sequence_length
                self.sequences.append(group_features[start_idx:end_idx])
                self.labels.append(group_labels[end_idx - 1])
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx:int) -> tuple[torch.tensor, torch.tensor]:
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)#.reshape(-1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sequence, label
    
    def print_examples(self, num_examples: int = 5):
        for i in range(min(num_examples, len(self))):
            sequence, label = self[i]
            print(f"Sequence {i + 1}:")
            print(sequence)
            print(f"Label: {label}\n")
