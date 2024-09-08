import matplotlib.pyplot as plt
import numpy as np


def plot_greyscale(arr: np.array):
    """Plot a figure for each image in the batch of arrays (arr).

    Parameter
    ---------
    arr : np.array
        A 4D array with shape (batch_size, height, width, channels).
    """
    for i in range(arr.shape[0]):
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.imshow(arr[i, :], cmap='Greys')
        plt.show()
