import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import torch
from typing import Tuple, List, Any

from utils.misc import random_select_elements, generate_random_indices, dec_to_sci

np.seterr(divide='ignore')


# -- Angle calculations method and associated utils functions -- #

def center_line(x1: float, y1: float, x2: float, y2: float) -> Tuple[float]:
    """
    Find the center of a line
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
    Get the point above the horizontal line passing through the center of the line between (x1,y1) and (x2,y2).
    :param x1: x position of first point
    :param y1: y position of first point
    :param x2: x position of second point
    :param y2: y position of second point
    :return: in order the point above the center and the point under the center
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

    a, b, c, d = get_point_above_horizontal(x1, y1, x2, y2)

    dx = a - c
    dy = b - d

    if dx == 0:
        return np.pi/2  # way to handle geometrically division by 0 in the formula of the slope
    else:
        slope = dy/dx
        angle = np.arctan(slope)
        if angle < 0:
            return angle + np.pi
        else:
            return angle


def normalize_angle(angle: Any):
    """
    Normalize angle in radian to a value between 0 and 1.
    Angle can be a float or a ndarray, it doesn't matter
    :param angle: angle of a line
    :return: normalized angle value
    """
    return angle / (2 * np.pi)


def angles_from_list(lines: List[Tuple[List]], normalize: bool = False) -> ndarray:
    """
    The list of lines contains tuples of list coordinate of the form ([x1, x2], [y1, y2]). It is a bother to calculate
    directly the angle using calculate_angle, so first we extract coordinates, and then apply the functions
    :param lines: List containing lines coordinates
    :param normalize: Whether to normalize the angles or not
    :return: List of angles associated with each line
    """
    # Initialise output list
    angle_list = []
    # Iterate through each line and find its angle
    for line_list in lines:
        line = line_list[0]  # when generated, the lines for each patch are in a list (of one element if you choose one intersecting line per patch)
        x1, x2 = line[0][0], line[0][1]
        y1, y2 = line[1][0], line[1][1]
        if normalize:
            angle = normalize_angle(calculate_angle(x1, y1, x2, y2))
        else:
            angle = calculate_angle(x1, y1, x2, y2)
        angle_list.append(angle)

    return np.array(angle_list)


""" 
Line rotation methods:

This is useful on single dot diagrams as only one orientation of line is possible. The goal is to expend the database 
with more angle values. However, it raises the issue of pixel modification, as not all rotation angles are allowed,
only multiples of 90°. It is recommended to work directly on double dots.
"""


def rotate_line(line, theta=np.pi/2):
    """
    Rotate line by an angle theta.
    Ref: https://www.reddit.com/r/askmath/comments/luimi0/equation_for_finding_the_line_end_point_positions/
    :param line:
    :param theta:
    :return:
    """
    x1, x2, y1, y2 = line[0][0], line[0][1], line[1][0], line[1][1]

    return [x1 * np.cos(theta) - y1 * np.sin(theta), x2 * np.cos(theta) - y2 * np.sin(theta)], [x1 * np.sin(theta) + y1 * np.cos(theta), x2 * np.sin(theta) + y2 * np.cos(theta)]


def random_choice_rotate(images_list, lines_list, nbr_to_rotate):
    """
    Randomly rotates image and lines by an angle choosen in [0,90,180,270].

    Tip: 'resample=Image.BILINEAR' argument is added to the F.rotate function. This argument specifies the resampling method used
    during rotation. Additionally, the 'expand=False' argument is provided to prevent the output image from being expanded
    to fit the rotated image entirely.
    :param images_list:
    :param lines_list:
    :param nbr_to_rotate: Number of images and lines to rotate
    :return:
    """
    rotated_images_list, rotated_lines_list = images_list.copy(), lines_list.copy()

    random_indices = generate_random_indices(len(images_list), nbr_to_rotate)
    for i in random_indices:
        # Select initial image and lines
        image = images_list[i]
        line = lines_list[i]

        # Rotate image and line by 90°
        rotated_image = torch.rot90(image)
        rotated_images_list[i] = rotated_image

        rotated_line = rotate_line(line)
        rotated_lines_list[i] = rotated_line

    return rotated_images_list, rotated_lines_list


# -- Line decomposition method -- #

def decompose_line(line) -> tuple[list[list[tuple[Any, Any]]], list[float]]:
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
    bin_size = 0.01
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
