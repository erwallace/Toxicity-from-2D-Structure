from toxic2d.transforms import SMILESToMol
from rdkit import Chem


def test_SMILESToMol():
    smile = "CCO"
    transformed = SMILESToMol().__call__(smile)

    assert isinstance(transformed, Chem.Mol)
