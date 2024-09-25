import torch
from rdkit import Chem
from src.transforms import MolToRDKitDescriptors


def test_MolToRDKitDescriptors():
    mol = Chem.MolFromSmiles("CCO")
    transformed = MolToRDKitDescriptors().__call__(mol)

    descriptors = transformed
    assert isinstance(descriptors, torch.Tensor)
    assert len(descriptors) == len(Chem.Descriptors.descList)
