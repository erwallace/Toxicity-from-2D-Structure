from rdkit import Chem
from src.transforms import MolToRDKitDescriptors


def test_MolToRDKitDescriptors():
    sample = tuple([Chem.MolFromSmiles("CCO"), 1])
    transformed = MolToRDKitDescriptors().__call__(sample)

    descriptors, toxicity = transformed
    assert isinstance(descriptors, list)
    assert len(descriptors) == len(Chem.Descriptors.descList)
    assert all(isinstance(d, (float, int)) for d in descriptors)
