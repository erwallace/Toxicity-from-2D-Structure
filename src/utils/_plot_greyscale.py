import matplotlib.pyplot as plt
import numpy as np


def plot_greyscale(arr: np.array):
    """Plot a figure for each image in the batch of arrays (arr).

    Parameter
    ---------
    arr : np.array
        A 4D array with shape (batch_size, height, width, channels).
        OR
        A 3D array with shape (height, width, channels).
    """

    def greyscale(arr):
        """Plot a greyscale figure.

        Parameter
        ---------
        arr : np.array
            A 3D array with shape (height, width, channels).
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.imshow(arr, cmap="Greys")
        plt.show()

    assert arr.shape[-1] == 1, "The array must have only one channel."

    if arr.dim() == 3:
        greyscale(arr)

    elif arr.dim() == 4:
        for i in range(arr.shape[0]):
            greyscale(arr[i, :])

    else:
        raise ValueError(
            f"The array has {arr.dim()} dimensions ({arr.shape}). Must have 3 or 4."
        )
