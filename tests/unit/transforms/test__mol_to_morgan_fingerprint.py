import pytest
import torch
from rdkit import Chem

from toxic2d.transforms import MolToMorganFingerprints


@pytest.mark.skip(reason="Test not implemented")
@pytest.mark.parametrize(
    "radius, n_bits, includeChirality, use_counts",
    [
        (2, 256, True, True),
        (2, 256, True, False),
        (2, 256, False, True),
        (2, 256, False, False),
    ],
)
def test_MolToRDKitDescriptors():
    mol = Chem.MolFromSmiles("CCO")
    transformed = MolToMorganFingerprints().__call__(mol)

    fingerprint = transformed
    assert isinstance(fingerprint, torch.Tensor)
