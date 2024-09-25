import pandas as pd
from torch.utils.data import Dataset
from src.constants import TOXICITIES


class Tox21Base(Dataset):
    def __init__(self, csv_path, transform=None, target_transform=None):
        self.data = pd.read_csv(csv_path)
        self.remove_nan()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def remove_nan(self) -> None:
        self.data = self.data.dropna(subset="smiles")
        self.data = self.data.dropna(subset=TOXICITIES, how="all")
        self.data = self.data.reset_index(drop=True)
        self.data = self.data.fillna(0)
