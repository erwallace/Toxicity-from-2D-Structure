from rdkit import Chem
from typing import Any


class SMILESToMol:
    def __call__(self, sample: tuple[str, Any]) -> tuple[Chem.Mol, Any]:
        smile, toxicity = sample
        return Chem.MolFromSmiles(smile), toxicity
