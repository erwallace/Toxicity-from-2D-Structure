import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class MolToGreyscale:
    def __init__(self, embed: float = 20.0, res: float = 0.5):
        self.embed = embed
        self.res = res

    def __call__(self, mol: Chem.Mol) -> np.array:
        return mol_to_greyscale(mol, self.embed, self.res)


def mol_to_greyscale(mol: Chem.Mol, embed: float = 20.0, res: float = 0.5) -> np.array:
    """Convert the molecule to a greyscale image of its 2D structure.

    Modified code from Esben J. Bjerrum

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to convert.
    embed : float
        The size of the image.
    res : float
        The resolution of the image.

    Returns
    -------
    np.array
        The greyscale image of the molecule.
    """
    len_ = int(embed * 2 / res)
    AllChem.Compute2DCoords(mol)
    coords = mol.GetConformer(0).GetPositions()
    vect = np.zeros((len_, len_, 1))

    # Bonds first
    for i, bond in enumerate(mol.GetBonds()):
        bondorder = bond.GetBondTypeAsDouble()
        start_coords = coords[bond.GetBeginAtomIdx()]
        end_coords = coords[bond.GetEndAtomIdx()]
        frac = np.linspace(0, 1, int(1 / res * 2))

        for f in frac:
            c = f * start_coords + (1 - f) * end_coords
            idx = int(round((c[0] + embed) / res))
            idy = int(round((c[1] + embed) / res))
            vect[idx, idy, 0] = bondorder

    # Atoms second
    for i, atom in enumerate(mol.GetAtoms()):
        idx = int(round((coords[i][0] + embed) / res))
        idy = int(round((coords[i][1] + embed) / res))
        # Atomic number
        vect[idx, idy, 0] = atom.GetAtomicNum()

    return vect
