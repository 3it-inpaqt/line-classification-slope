import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
from matplotlib.figure import Figure


def create_multiplots(batch: ndarray):
    """
    Generate figures with several plots to see different lines orientation

    :param batch: size of the image
    :return: a figure with subplots
    """

    p, _, n = batch.shape
    # Compute the number of rows and columns required to display n subplots
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))

    # Create a figure and axis objects
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    plt.tight_layout()

    for i, ax in enumerate(axes.flatten()):
        if i < n:
            ax.imshow(batch[:, :, i], cmap='gray')
            ax.axis('off')
        else:
            fig.delaxes(ax) # if not there, problem with range in the array and out of bound error

    return fig, axes
