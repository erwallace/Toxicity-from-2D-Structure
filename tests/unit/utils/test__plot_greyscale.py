import numpy as np
import pytest

from toxic2d.utils import plot_greyscale


@pytest.mark.parametrize(
    "arr, should_raise, exception",
    [
        (np.random.rand(10, 10, 1), False, None),  # Valid 3D array
        (np.random.rand(2, 10, 10, 1), False, None),  # Valid 4D array
        (
            np.random.rand(10, 10, 3),
            True,
            AssertionError,
        ),  # Invalid 3D array: more than one channel
        (
            np.random.rand(10, 1),
            True,
            ValueError,
        ),  # Invalid 2D array: wrong number of dimensions
    ],
)
def test_plot_greyscale(arr, should_raise, exception):
    if should_raise:
        with pytest.raises(exception):
            plot_greyscale(arr)
    else:
        plot_greyscale(arr)
