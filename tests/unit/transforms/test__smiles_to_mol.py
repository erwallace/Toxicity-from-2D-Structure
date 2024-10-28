from rdkit import Chem

from toxic2d.transforms import SMILESToMol


def test_SMILESToMol():
    smile = "CCO"
    transformed = SMILESToMol().__call__(smile)

    assert isinstance(transformed, Chem.Mol)
