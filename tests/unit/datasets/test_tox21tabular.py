import pytest

from toxic2d.datasets import Tox21Tabular


@pytest.mark.skip(reason="Test not implemented")
def test_Tox21Tabular(csv_path):
    tox21 = Tox21Tabular(csv_path)
    return tox21
