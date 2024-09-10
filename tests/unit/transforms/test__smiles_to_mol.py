from src.transforms import SMILESToMol
from rdkit import Chem


def test_SMILESToMol():
    sample = tuple(["COC", None])
    transformed = SMILESToMol().__call__(sample)

    assert isinstance(transformed[0], Chem.Mol)
