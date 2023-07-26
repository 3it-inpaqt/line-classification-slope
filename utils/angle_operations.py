import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import torch
from typing import Tuple, List, Any

from utils.settings import settings

np.seterr(divide='ignore')


# -- Angle calculations method and associated utils functions -- #

def center_line(x1: float, y1: float, x2: float, y2: float) -> Tuple[float]:
    """
    Find the center of a line based on its endpoints coordinates
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return: Tuple center x and y coordinates
    """
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y


def get_point_above_horizontal(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """
    Get the point above the horizontal line passing through the center of the line between (x1,y1) and (x2,y2). This is
    to get the line to follow a symmetry rule (180° -> 0°).
    :param x1: x position of first point
    :param y1: y position of first point
    :param x2: x position of second point
    :param y2: y position of second point
    :return: In order the point above the center and the point under the center
    """
    # Calculate the y-center of the line
    center_y = (y1 + y2) / 2

    if y1 >= center_y:
        return x1, y1, x2, y2
    else:
        return x2, y2, x1, y1


def calculate_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the angle of a line
    :param x1: x position of first point
    :param y1: y position of first point
    :param x2: x position of second point
    :param y2: y position of second point
    :return: angle of a line between (x1,y1) and (x2,y2) with respect to the x-axis
    """
    # Re-order the coordinates of the line to ensure the angle is between 0° and 180° later on
    a, b, c, d = get_point_above_horizontal(x1, y1, x2, y2)

    dx = a - c
    dy = b - d

    if dx == 0:
        return np.pi/2  # way to geometrically handle division by 0 in the formula of the slope because of tan function
    else:
        slope = dy/dx
        angle = np.arctan(slope)
        # If the angle is negative, simply add pi to take its value on the other side of the trigonometric circle
        if angle < 0:
            return angle + np.pi
        # Otherwise the angle is already between 0° and 180°
        else:
            return angle


def calculate_angle_full_circle(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the angle of a line within the range [0°, 360°] (no symmetry)
    :param x1: x position of first point
    :param y1: y position of first point
    :param x2: x position of second point
    :param y2: y position of second point
    :return: angle of a line between (x1,y1) and (x2,y2) with respect to the x-axis
    """

    a, b, c, d = get_point_above_horizontal(x1, y1, x2, y2)

    dx = a - c
    dy = b - d

    if dx == 0:
        return np.pi/2  # way to handle geometrically division by 0 in the formula of the slope
    else:
        slope = dy/dx
        angle = np.arctan(slope) % (2 * np.pi)
        return angle


def normalize_angle(angle: Any) -> Any:
    """
    Normalize angle in radian to a value between 0 and 1.
    Angle can be a float or a ndarray, it doesn't matter
    :param angle: angle of a line
    :return: normalized angle value
    """
    return angle / (2 * np.pi)


def angle_from_line(line: List[Tuple[List]], normalize: bool = False) -> ndarray:
    """
    Find angle of one single line item [([x1, x2], [y1, y2])]
    :param line:
    :param normalize:
    :return:
    """
    # Get the line coordinates from the list object
    x1, x2 = line[0][0][0], line[0][0][1]
    y1, y2 = line[0][1][0], line[0][1][1]

    if settings.full_circle:
        angle = calculate_angle_full_circle(x1, y1, x2, y2)  # don't take the symmetry into account
    else:
        angle = calculate_angle(x1, y1, x2, y2)  # take the symmetry into account

    if normalize:
        angle = normalize_angle(angle)  # normalize angle values by 2 pi

    return angle


def angles_from_list(lines: List[Any], normalize: bool = False) -> ndarray:
    """
    The list of lines contains tuples of list coordinate of the form ([x1, x2], [y1, y2]). It is a bother to calculate
    directly the angle using calculate_angle, so first we extract coordinates, and then apply the functions.
    :param lines: List containing lines coordinates
    :param normalize: Whether to normalize the angles or not
    :return: List of angles associated with each line
    """
    # Initialise output list
    angle_list = []
    # Iterate through each line and find its angle
    for line_list in lines:
        line = line_list[0]  # when generated, the lines for each patch are in a list (of one element if you choose one intersecting line per patch)
        # print(line_list)
        x1, x2 = line[0][0], line[0][1]
        y1, y2 = line[1][0], line[1][1]

        if settings.full_circle:
            angle = calculate_angle_full_circle(x1, y1, x2, y2)
        else:
            angle = calculate_angle(x1, y1, x2, y2)

        if normalize:
            angle = normalize_angle(angle)

        angle_list.append(angle)

    return np.array(angle_list)


# -- Line decomposition method -- #

def decompose_line(line: List[Any]) -> Tuple[List[List[Tuple[Any, Any]]], List[float]]:
    """
    Find angle of all the lines composing a single segment. Sometimes, a single line passes through a patch, but is not
    perfectly straight due to the labeling and/or setup variability.
    :param line:
    :return:
    """
    x_line = line[0]
    y_line = line[1]

    decomposition = []
    angles_decomposition = []
    for i in range(0, len(x_line), 2):
        for x1, x2, y1, y2 in zip(x_line[i:i + 2], x_line[i + 1:i + 3], y_line[i:i + 2],
                                  y_line[i + 1:i + 3]):
            decomposition.append([(x1, x2), (y1, y2)])
            angles_decomposition.append(calculate_angle(x1, y1, x2, y2))

    return decomposition, angles_decomposition


# -- STATISTICS ON ANGLE -- #

def get_angle_stat(angles_list: List[float]) -> None:
    """
    Get angles distribution of the dataset.
    :param angles_list:
    :return:
    """
    fig, ax = plt.subplots()

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.subplots_adjust(bottom=0.15)
    # Adjust the bin size
    bin_size = 0.005
    len_list = len(angles_list)
    avg = round(sum(angles_list)/len_list, 3)

    # Add a text box to the plot
    textstr = r'$Average: {{{avg}}}$'.format(avg=avg, )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.1, 0.9, textstr, transform=ax.transAxes, fontsize=14, ha='left', va='top', bbox=props)

    # Plot histogram
    plt.hist(angles_list, bins=int((max(angles_list) - min(angles_list)) / bin_size), density=True)
    plt.xlabel(r'Angles', fontsize=18)
    plt.ylabel(r'Frequency (\%)', fontsize=18)
    plt.show()
