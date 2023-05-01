import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
from utils.angleoperations import calculate_angle
from matplotlib.figure import Figure


def create_multiplots(batch: ndarray, angle_list: list[float]):
    """
    Generate figures with several plots to see different lines orientation

    :param batch: size of the image
    :return: a figure with subplots
    """

    n, p, _ = batch.shape
    # Compute the number of rows and columns required to display n subplots
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))

    # Create a figure and axis objects
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    plt.tight_layout()

    for i, ax in enumerate(axes.flatten()):
        if i < n:
            image = batch[i, :, :]
            angle_radian = angle_list[i]
            angle_degree = np.rad2deg(angle_radian)
            ax.imshow(image, cmap='gray')
            ax.set_title('Angle: {:.2f} | {:.2f}°'.format(angle_radian, angle_degree))
            ax.axis('off')
        else:
            fig.delaxes(ax) # if not there, problem with range in the array and out of bound error

    return fig, axes
