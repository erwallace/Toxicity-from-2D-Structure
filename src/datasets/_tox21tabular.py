from sklearn.preprocessing import StandardScaler
from src.datasets import Tox21Base
from src.constants import TOXICITIES
import numpy as np
import torch


class Tox21Tabular(Tox21Base):
    def __init__(self, csv_path, transform=None, target_transform=None):
        super().__init__(csv_path, transform, target_transform)

        # ToDo: is there a better way of doing this? even with only 7,000 datapoints its slow
        # Apply transformations to the entire dataset if transform is provided
        if self.transform:
            transformed_smiles = [
                self.transform(smile) for smile in self.data["smiles"]
            ]
            self._features = torch.stack(transformed_smiles).numpy()
        else:
            self._features = self.data["smiles"].values
        # Fit and transform the data using StandardScaler
        nan_indices = np.where(np.isnan(self._features).any(axis=1))[0]
        self.scaler = StandardScaler().fit(self._features[~nan_indices])

        # Remove any NaN rows that may have been introduced by the transform
        print(f"Removing {len(nan_indices)} NaN rows from the dataset")
        self.data = self.data.drop(nan_indices).reset_index(drop=True)

    def __getitem__(self, idx: int) -> tuple:
        smile = self.data["smiles"].iloc[idx]
        toxicity = self.data[TOXICITIES].iloc[idx].values

        if self.transform:
            features = self.transform(smile)
        if self.target_transform:
            target = self.target_transform(toxicity)

        features = self.scaler.transform(
            features.reshape(1, -1)
        ).flatten()  # reshaped for a single sample

        return (torch.Tensor(features), target)
