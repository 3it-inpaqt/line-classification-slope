from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from typing import Tuple, Any

from utils.settings import settings


def create_multiplots(image_set_input: Any, angles: Any, prediction_angles: Any = None, number_sample: float = None) -> Tuple[Figure, Axes]:
    """
    Generate figures with several plots to see different lines orientation

    :param image_set_input:
    :param angles: array containing the angles for each image of the set
    :param prediction_angles: optional, value of predicted angles by a neural network (ndarray)
    :param number_sample: number of images to plot, None by default
    :return: a figure with subplots
    """
    if settings.synthetic:  # for synthetic diagrams
        n = image_set_input.shape[0]
        p, q = settings.patch_size_x, settings.patch_size_y
        image_set = image_set_input.reshape(n, p, q)

    else:  # for experimental diagrams
        image_set = image_set_input.squeeze(1)  # tensor of shape [n, N*N] required
        n, _ = image_set.shape
        image_set.reshape(n, settings.patch_size_x, settings.patch_size_y)

    # Make sure the program doesn't sample more data than available
    if (number_sample is not None) and (number_sample > n):
        number_sample = n // 2  # choose arbitrary half the dataset

    # Compute the number of rows and columns required to display n subplots
    number_rows = int(np.ceil(np.sqrt(n)))
    number_columns = int(np.ceil(n / number_rows))

    # Select a random sample of indices
    indices = sample(range(number_sample), k=number_sample)

    # Create a figure and axis objects
    fig, axes = plt.subplots(nrows=number_rows, ncols=number_columns, figsize=(6 * number_columns, 6 * number_rows))

    for i, ax in enumerate(axes.flatten()):
        if i < n:
            # Get the patch
            index = indices[i]
            image = image_set[index, :, :]  # no need to reshape, image is now [N, N] shape
            # Get the angle
            normalized_angle = float(angles[index])
            angle_radian = normalized_angle * (2 * np.pi)
            angle_degree = angle_radian * 180 / np.pi
            # Set the figure
            ax.imshow(image * 255, cmap='copper')  # first show the image otherwise line would be hidden
            title = 'Angle: {:.3f} | {:.2f}° \n Normalized value: {:.4f}'.format(angle_radian, angle_degree, normalized_angle)
            # Modify figure title to take into account predicted angle value if it was given in input
            if prediction_angles is not None:
                prediction_angle = prediction_angles[index][0]  # the angle is a ndarray type with one element only for index i
                title += '\n Predicted: {:.4f} ({:.2f}°)'.format(prediction_angle, prediction_angle*2*np.pi*180/np.pi)
            # Set ax properties
            ax.set_title(title, fontsize=25)
            ax.axis('off')
            plt.tight_layout()

        else:
            # Removes empty axis if number of patches is not a perfect square (not 9, 16, 25, ...)
            fig.delaxes(ax)  # if not there, problem with range in the array and out of bound error

    return fig, axes
