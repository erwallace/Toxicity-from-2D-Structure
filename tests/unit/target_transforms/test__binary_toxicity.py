from src.target_transforms import BinaryToxicity
import pytest


@pytest.mark.parametrize(
    "toxicity_array, binary_output",
    [
        ([0, 1, 1], 1),
        ([0, 0], 0),
    ],
)
def test_BinaryToxicity(toxicity_array, binary_output):
    sample = tuple(["smile", toxicity_array])
    transformed = BinaryToxicity().__call__(sample)

    assert transformed[1] == binary_output
