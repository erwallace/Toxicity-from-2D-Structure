from toxic2d.transforms import MolToGreyscale
from toxic2d.transforms._mol_to_greyscale import mol_to_greyscale
import numpy as np
from rdkit import Chem


def test_mol_to_greyscale_valid_molecule():
    mol = Chem.MolFromSmiles("CCO")
    result = mol_to_greyscale(mol, embed=20, res=0.5)

    assert isinstance(result, np.ndarray)
    assert result.shape == (80, 80, 1)


def test_MolToGreyscale():
    mol = Chem.MolFromSmiles("CCO")
    transformed = MolToGreyscale(embed=20, res=0.5).__call__(mol)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (80, 80, 1)
