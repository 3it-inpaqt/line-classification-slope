from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from typing import Tuple, Any

from utils.settings import settings


def create_multiplots(image_set_input: Any, angles: Any, prediction_angles: Any = None, number_sample: float = None, cmap: str = 'copper', normalize=True) -> Tuple[Figure, Axes]:
    """
    Generate figures with several plots to see different lines orientation

    :param image_set_input:
    :param angles: array containing the angles for each image of the set
    :param prediction_angles: optional, value of predicted angles by a neural network (ndarray)
    :param number_sample: number of images to plot, None by default
    :param cmap: Color map, copper by default
    :param normalize: Whether the angles are normalized or not, True by default
    :return: a figure with subplots
    """
    if settings.synthetic:  # for synthetic diagrams
        n = image_set_input.shape[0]
        p, q = settings.patch_size_x, settings.patch_size_y
        image_set = image_set_input.reshape(n, p, q)

    else:  # for experimental diagrams
        image_set = image_set_input.squeeze(1)  # tensor of shape [n, N*N] required
        # print(image_set.shape)
        if settings.model_type == 'FF':
            print(image_set.shape)
            n, _ = image_set.shape
            image_set = image_set.reshape(n, settings.patch_size_x, settings.patch_size_y)

        elif settings.model_type == 'CNN':
            n, _, _ = image_set.shape

    # Make sure the program doesn't sample more data than available
    if (number_sample is not None) and (number_sample > n):
        number_sample = n // 2  # choose arbitrary half the dataset

    # Compute the number of rows and columns required to display n subplots
    number_rows = int(np.ceil(np.sqrt(number_sample)))
    number_columns = int(np.ceil(number_sample / number_rows))

    # Select a random sample of indices
    indices = sample(range(number_sample), k=number_sample)

    # Create a figure and axis objects
    fig, axes = plt.subplots(nrows=number_rows, ncols=number_columns, figsize=(4 * number_columns, 4 * number_rows))

    # print(image_set.shape)

    for i, ax in enumerate(axes.flatten()):
        if i < number_sample:
            # Get the patch
            index = indices[i]
            image = image_set[index, :, :]  # no need to reshape, image is now [N, N] shape
            # Get the angle
            angle_radian = float(angles[index])
            if normalize:
                angle_radian = angle_radian * (2 * np.pi)
            angle_degree = angle_radian * 180 / np.pi
            # Set the figure
            ax.imshow(image * 255, cmap=cmap)  # first show the image otherwise line would be hidden
            title = 'Angle: {:.2f} ({:.2f}°)'.format(angle_radian, angle_degree)
            # Modify figure title to take into account predicted angle value if it was given in input
            if prediction_angles is not None:
                # print(prediction_angles)
                prediction_angle = prediction_angles[index][0]  # the angle is a ndarray type with one element only for index i
                if normalize:
                    prediction_angle_degree = prediction_angle * 2 * np.pi * 180 / np.pi
                else:
                    prediction_angle_degree = prediction_angle * 180 / np.pi
                title += '\n Predicted: {:.2f} ({:.2f}°)'.format(prediction_angle, prediction_angle_degree)
            # Set ax properties
            ax.set_title(title, fontsize=22)
            ax.axis('off')
            plt.tight_layout()

        else:
            # Removes empty axis if number of patches is not a perfect square (not 9, 16, 25, ...)
            fig.delaxes(ax)  # if not there, problem with range in the array and out of bound error

    return fig, axes
