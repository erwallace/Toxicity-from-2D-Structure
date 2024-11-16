import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from toxic2d.constants import TOXICITIES
from toxic2d.datasets import Tox21Base


class Tox21Tabular(Tox21Base):
    def __init__(self, csv_path, transform=None, target_transform=None):
        super().__init__(csv_path, transform, target_transform)
        self.scaler = None
        self.transformed_features = None
        self.scaler_fit()

    def __getitem__(self, idx: int) -> tuple:
        features = self.data["smiles"].iloc[idx]
        label = self.data[TOXICITIES].iloc[idx].values

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        features = self.scaler_transform(features)

        return features, label

    def scaler_fit(self):
        """Fits the StandardScaler to the dataset.

        To do this, we first apply the transformation to the entire dataset, remove any
        nan rows that the transformations create and then fit the scaler to the
        transformed data.
        """
        # ToDo: is there a better way of doing this? even with only 7,000 datapoints its slow
        if self.transform:
            transformed_smiles = [self.transform(smile) for smile in self.data["smiles"]]
            self.transformed_features = torch.stack(transformed_smiles).numpy()

            nan_idxs = np.where(np.isnan(self.transformed_features).any(axis=1))[0]
            self.scaler = StandardScaler().fit(self.transformed_features[~nan_idxs])

            print(f"Removing {len(nan_idxs)} NaN rows from the dataset")
            self.data = self.data.drop(nan_idxs).reset_index(drop=True)

    def scaler_transform(self, features):
        """Uses sklearn scaler to transform data. Must be called after scale_fit."""
        if self.scaler is None:
            raise ValueError("Scaler has not been fit. Please run scaler_fit() first.")
        features = self.scaler.transform(features.reshape(1, -1))  # reshaped for a single sample
        features = torch.Tensor(features.flatten())
        return features
