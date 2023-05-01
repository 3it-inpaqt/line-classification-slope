from typing import Tuple

import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from utils.angleoperations import normalize_angle

def create_multiplots(image_set: ndarray, angles: ndarray) -> Tuple[Figure, Axes]:
    """
    Generate figures with several plots to see different lines orientation

    :param image_set: size of the image
    :param angles: array containing the angles for each image of the set
    :return: a figure with subplots
    """

    n, p, _ = image_set.shape
    # Compute the number of rows and columns required to display n subplots
    number_rows = int(np.ceil(np.sqrt(n)))
    number_columns = int(np.ceil(n / number_rows))

    # Create a figure and axis objects
    fig, axes = plt.subplots(nrows=number_rows, ncols=number_columns, figsize=(4 * number_columns, 4 * number_rows))
    plt.tight_layout()

    for i, ax in enumerate(axes.flatten()):
        if i < n:
            image = image_set[i, :, :]
            angle_radian = angles[i]
            angle_degree = np.rad2deg(angle_radian)
            normalized_angle = normalize_angle(angle_radian)
            ax.imshow(image, cmap='gray')
            ax.set_title('Angle: {:.2f} | {:.2f}° \n Normalized value: {:.2f}'.format(angle_radian, angle_degree, normalized_angle), fontsize=20)
            ax.axis('off')
        else:
            fig.delaxes(ax)  # if not there, problem with range in the array and out of bound error

    return fig, axes
