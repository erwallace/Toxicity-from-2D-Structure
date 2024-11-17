import pytest

from toxic2d import test_dir


@pytest.fixture(scope="session")
def csv_path():
    return test_dir / "data" / "tox21_sample.csv"
