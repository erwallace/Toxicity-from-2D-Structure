from rdkit import Chem
from typing import Any


class SMILESToMol:
    def __call__(self, feature: str) -> tuple[Chem.Mol, Any]:
        return Chem.MolFromSmiles(feature)
