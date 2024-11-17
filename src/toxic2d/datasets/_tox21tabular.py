from collections.abc import Callable

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from toxic2d.constants import TOXICITIES
from toxic2d.datasets import Tox21Base


class Tox21Tabular(Tox21Base):
    def __init__(
        self,
        csv_path: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        scaling: bool = True,
        remove_tranformed_nan: bool = True,
    ):
        super().__init__(csv_path, transform, target_transform)
        self.transformed_features = None
        self.scaling = scaling

        if scaling and not remove_tranformed_nan:
            raise ValueError("NaN removal must be enabled when scaling is enabled.")

        if remove_tranformed_nan:
            self.remove_tranformed_nan()

        if self.scaling:
            self.scaler = None
            self.scaler_fit()

    def __getitem__(self, idx: int) -> tuple:
        features = self.data["smiles"].iloc[idx]
        label = self.data[TOXICITIES].iloc[idx].values

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        if self.scaling:
            features = self.scaler_transform(features)

        return features, label

    def remove_tranformed_nan(self):
        """Removes any rows that contain NaN values after the transformation."""
        # TODO: is there a better way of doing this? even with only 7,000 datapoints its slow
        if self.transform:
            transformed_smiles = [self.transform(smile) for smile in self.data["smiles"]]
            self.transformed_features = torch.stack(transformed_smiles).numpy()

            # Replace remove np.inf and np.nan rows
            self.transformed_features[np.isinf(self.transformed_features)] = np.nan
            nan_idxs = np.where(np.isnan(self.transformed_features).any(axis=1))[0]
            self.transformed_features = self.transformed_features[~nan_idxs]

            print(f"Removing {len(nan_idxs)} transformed NaN rows from the dataset")
            self.data = self.data.drop(nan_idxs).reset_index(drop=True)

    def scaler_fit(self):
        """Fits the StandardScaler to the dataset. Must be called after remove_tranformed_nan()."""
        self.scaler = StandardScaler().fit(self.transformed_features)

    def scaler_transform(self, features):
        """Uses sklearn scaler to transform data. Must be called after scaler_fit()."""
        if self.scaler is None:
            raise ValueError("Scaler has not been fit. Please run scaler_fit() first.")
        features = self.scaler.transform(features.reshape(1, -1))  # reshaped for a single sample
        features = torch.Tensor(features.flatten())
        return features
