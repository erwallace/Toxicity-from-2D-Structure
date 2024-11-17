import pandas as pd

from toxic2d.datasets import Tox21Base


def test__tox21tabular(csv_path):
    csv = pd.read_csv(csv_path)
    assert len(csv) == 10

    tox21 = Tox21Base(csv_path=csv_path)
    assert len(tox21) == 7

    assert tox21.data.isna().sum().sum() == 0
