import numpy as np
import pytest
from torch.utils.data import DataLoader
from torchvision import transforms

from toxic2d.datasets import Tox21Base, Tox21Tabular
from toxic2d.target_transforms import BinaryToxicity
from toxic2d.transforms import MolToRDKitDescriptors, SMILESToMol


def test_Tox21Tabular_remove_transformed_nan(csv_path):
    tox21_base = Tox21Base(
        csv_path, transform=transforms.Compose([SMILESToMol(), MolToRDKitDescriptors()])
    )
    assert len(tox21_base) == 7

    tox21 = Tox21Tabular(
        csv_path,
        transform=transforms.Compose([SMILESToMol(), MolToRDKitDescriptors()]),
        scaling=False,
        remove_tranformed_nan=True,
    )
    # idx = 9 is removed from df because it has some NaN rdkit descriptors
    assert len(tox21) == 6


def test_Tox21Tabular_scaler_fit(csv_path):
    tox21 = Tox21Tabular(
        csv_path,
        transform=transforms.Compose([SMILESToMol(), MolToRDKitDescriptors()]),
        scaling=True,
        remove_tranformed_nan=True,
    )
    scaled = tox21.scaler.transform(tox21.transformed_features)

    assert np.isclose(scaled.mean(axis=0).max(), 0)
    assert np.isclose(scaled.mean(axis=0).min(), 0)
    assert np.isclose(scaled.std(axis=0).max(), 1)
    assert np.isclose(scaled.std(axis=0).min(), 1)


def test_Tox21Tabular_scaler_fit_value_error(csv_path):
    with pytest.raises(ValueError):
        Tox21Tabular(csv_path, scaling=True, remove_tranformed_nan=False)


@pytest.mark.skip(reason="Feature scaling is not working currently")
def test_Tox21Tabular_scaler_transform(csv_path):
    tox21 = Tox21Tabular(
        csv_path=csv_path,
        transform=transforms.Compose([SMILESToMol(), MolToRDKitDescriptors()]),
        target_transform=BinaryToxicity(),
        remove_tranformed_nan=True,
        scaling=True,
    )
    dataloader = DataLoader(tox21, batch_size=100, shuffle=True)
    dataloader_iter = iter(dataloader)
    batch = next(dataloader_iter)
    features, _ = batch

    print(features.std(axis=0).max())
    print(features.std(axis=0).min())
    assert np.isclose(features.mean(axis=0).max(), 0, atol=1e-6)
    assert np.isclose(features.mean(axis=0).min(), 0, atol=1e-6)
    assert np.isclose(features.std(axis=0).max(), 1, atol=1e-6)
    assert np.isclose(features.std(axis=0).min(), 1, atol=1e-6)
