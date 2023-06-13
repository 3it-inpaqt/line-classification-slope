from typing import Tuple

import matplotlib.pyplot as plt
from numpy import ndarray
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from random import sample

from utils.angle_operations import normalize_angle
from utils.settings import Settings


def create_multiplots(image_set_tensor: ndarray, angles: ndarray, prediction_angles: ndarray = None, number_sample: float = None) -> Tuple[Figure, Axes]:
    """
    Generate figures with several plots to see different lines orientation

    :param image_set_tensor:
    :param angles: array containing the angles for each image of the set
    :param prediction_angles: optional, value of predicted angles by a neural network (ndarray)
    :param number_sample: number of images to plot, None by default
    :return: a figure with subplots
    """

    image_set = image_set_tensor.squeeze(1)
    n, p, _ = image_set.shape
    # n, p = image_set.shape  # change when using tensor
    # print(len(image_set))
    # n = len(image_set)  # change when using synthetic data

    if (number_sample is not None) and (number_sample < n):
        n = number_sample

    # Compute the number of rows and columns required to display n subplots
    number_rows = int(np.ceil(np.sqrt(n)))
    number_columns = int(np.ceil(n / number_rows))

    # Select a random sample of indices
    indices = sample(range(len(image_set)), k=number_sample)

    # Create a figure and axis objects
    fig, axes = plt.subplots(nrows=number_rows, ncols=number_columns, figsize=(6 * number_columns, 6 * number_rows))

    for i, ax in enumerate(axes.flatten()):
        if i < n:
            index = indices[i]
            # image = np.reshape(image_set[index, :, :], (Settings.patch_size_x, Settings.patch_size_y))
            image = image_set[index, :, :]

            normalized_angle = float(angles[index])
            # print(normalized_angle)
            angle_radian = normalized_angle * (2 * np.pi)
            # print(angle_radian)
            angle_degree = angle_radian * 180 / np.pi
            ax.imshow(image, cmap='copper')
            title = 'Angle: {:.3f} | {:.2f}° \n Normalized value: {:.4f}'.format(angle_radian, angle_degree, normalized_angle)
            if prediction_angles is not None:
                prediction_angle = prediction_angles[index][0]  # the angle is a ndarray type with one element only for index i
                title += '\n Predicted: {:.4f} ({:.2f}°)'.format(prediction_angle, prediction_angle*2*np.pi*180/np.pi)
            ax.set_title(title, fontsize=25)
            ax.axis('off')
            plt.tight_layout()
        else:
            fig.delaxes(ax)  # if not there, problem with range in the array and out of bound error

    return fig, axes
