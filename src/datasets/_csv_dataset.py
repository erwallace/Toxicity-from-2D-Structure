import pandas as pd
from torch.utils.data import Dataset
from src.constants import TOXICITIES
import numpy as np


class Tox21Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[str, np.array]:
        smile = self.data["smiles"].iloc[idx]
        toxicity = self.data[TOXICITIES].iloc[idx].values
        sample = (smile, toxicity)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def remove_nan(self) -> None:
        self.data = self.data.dropna(subset="smiles")
        self.data = self.data.dropna(subset=TOXICITIES, how="all")
        self.data = self.data.reset_index(drop=True)
        self.data = self.data.fillna(0)
