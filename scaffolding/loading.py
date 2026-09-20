import torch

from torch.utils.data import Dataset, DataLoader

class StoriesData(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path

        with open(file_path, encoding='utf-8') as f:
            self.raw_data = list(map(lambda s: s.strip(), f.readlines()))
            self.data = []


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            return IndexError(f"Index {idx} out of range.")
        return self.data[idx].strip()


