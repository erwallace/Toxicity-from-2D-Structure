import pytest

from toxic2d.target_transforms import BinaryToxicity


@pytest.mark.parametrize(
    "toxicity_array, binary_output",
    [
        ([0, 1, 1], 1),
        ([0, 0], 0),
    ],
)
def test_BinaryToxicity(toxicity_array, binary_output):
    transformed = BinaryToxicity().__call__(toxicity_array)

    assert transformed == binary_output
