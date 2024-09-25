import torch
from rdkit import Chem
from rdkit.Chem import Descriptors


class MolToRDKitDescriptors:
    def __call__(self, mol: Chem.Mol) -> torch.Tensor:
        funcs = [func for name, func in Descriptors.descList]
        desc = [func(mol) for func in funcs]
        return torch.Tensor(desc)
