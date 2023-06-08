from typing import Tuple

import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from utils.angle_operations import normalize_angle
from utils.settings import Settings


def create_multiplots(image_set: ndarray, angles: ndarray, prediction_angles: ndarray = None, number_sample: float = None) -> Tuple[Figure, Axes]:
    """
    Generate figures with several plots to see different lines orientation

    :param image_set: size of the image
    :param angles: array containing the angles for each image of the set
    :param prediction_angles: optional, value of predicted angles by a neural network (ndarray)
    :param number_sample: number of images to plot, None by default
    :return: a figure with subplots
    """

    # n, p, _ = image_set.shape
    n, p = image_set.shape
    if (number_sample is not None) and (number_sample < n):
        n = number_sample
    # Compute the number of rows and columns required to display n subplots
    number_rows = int(np.ceil(np.sqrt(n)))
    number_columns = int(np.ceil(n / number_rows))

    # Create a figure and axis objects
    fig, axes = plt.subplots(nrows=number_rows, ncols=number_columns, figsize=(4 * number_columns, 4 * number_rows))
    plt.tight_layout()

    for i, ax in enumerate(axes.flatten()):
        if i < n:
            image = np.reshape(image_set[i, :], (Settings.patch_size_x, Settings.patch_size_y))
            normalized_angle = float(angles[i])
            print(normalized_angle)
            angle_radian = normalized_angle * (2 * np.pi)
            print(angle_radian)
            angle_degree = angle_radian * 180 / np.pi
            ax.imshow(image, cmap='copper')
            title = 'Angle: {:.3f} | {:.3f}° \n Normalized value: {:.3f}'.format(angle_radian, angle_degree, normalized_angle)
            if prediction_angles is not None:
                prediction_angle = prediction_angles[i][0]  # the angle is a ndarray type with one element only for index i
                title += '\n Predicted value: {:.3f}'.format(prediction_angle)
            ax.set_title(title, fontsize=20)
            ax.axis('off')
        else:
            fig.delaxes(ax)  # if not there, problem with range in the array and out of bound error

    return fig, axes
