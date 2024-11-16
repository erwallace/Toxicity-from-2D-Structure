from rdkit import Chem


class SMILESToMol:
    def __call__(self, smile: str) -> Chem.Mol:
        return Chem.MolFromSmiles(smile)
