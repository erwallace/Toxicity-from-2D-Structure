from rdkit import Chem
from typing import Any
import numpy as np


class MolToRDKitDescriptors:
    def __call__(self, sample: tuple[Chem.Mol, Any]) -> tuple[np.array, Any]:
        mol, toxicity = sample
        return mol_to_rdkit_descriptors(mol), toxicity


def mol_to_rdkit_descriptors(mol: Chem.Mol) -> np.array:
    # TODO: Implement this function
    pass
