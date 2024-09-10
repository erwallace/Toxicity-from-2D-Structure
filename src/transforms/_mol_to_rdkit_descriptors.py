from rdkit import Chem
from typing import Any
import numpy as np
from rdkit.Chem import Descriptors


class MolToRDKitDescriptors:
    def __call__(self, sample: tuple[Chem.Mol, Any]) -> tuple[np.array, Any]:
        mol, toxicity = sample
        funcs = [func for name, func in Descriptors.descList]
        desc = [func(mol) for func in funcs]
        return desc, toxicity
