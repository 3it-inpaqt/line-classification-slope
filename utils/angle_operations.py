import numpy as np
from numpy import ndarray
import random

import torch
import torchvision.transforms.functional as f
from PIL import Image

from utils.misc import random_select_elements, generate_random_indices

from typing import Tuple, List
np.seterr(divide='ignore')


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
    Calculate the angle of a lign with respect to the
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
        return np.pi/2
    else:
        slope = dy/dx
        angle = np.arctan(slope)
        if angle < 0:
            return angle + np.pi
        else:
            return angle


def normalize_angle(angle):
    """
    Normalize angle in radian to a value between 0 and 1
    angle can be a float or a ndarray, it doesn't matter
    :param angle: angle of a line
    :return: normalized angle value
    """
    return angle / (2*np.pi)


def angles_from_list(lines: List[Tuple[List]]) -> ndarray:
    """
    The list of lines contains tuples of list coordinate of the form ([x1, x2], [y1, y2]). It is a bother to calculate
    directly the angle using calculate_angle, so first we extract coordinates, and then apply the functions
    :param lines: List containing lines coordinates
    :return: List of angles associated with each line
    """
    angle_list = []
    for line in lines:
        x1, x2 = line[0][0], line[0][1]
        y1, y2 = line[1][0], line[1][1]
        angle = calculate_angle(x1, y1, x2, y2)
        angle_list.append(angle)

    return np.array(angle_list)


def rotate_line(line, angle):
    """
    Rotate a line by an angle given by the user. Make sure the angle is in degree.
    :param line:
    :param angle:
    :return: Rotated line
    """

    # Extract line coordinates
    x1, x2 = line[0]
    y1, y2 = line[1]

    angle_rad = np.deg2rad(angle)

    # Create a 2x2 rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Apply rotation to line coordinates
    rotated_line = np.dot(rotation_matrix, np.array([[x1, x2], [y1, y2]]))
    # Extract rotated line coordinates
    rotated_x1, rotated_x2 = rotated_line[0]
    rotated_y1, rotated_y2 = rotated_line[1]

    return [rotated_x1, rotated_x2], [rotated_y1, rotated_y2]


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
    # from utils.misc import generate_random_angle
    #
    # angles = [generate_random_angle() for _ in range(len(images_list))]
    # print(images_list[0].size())
    # rotated_images_list = [f.rotate(image.permute(1, 2, 0), angle, interpolation=Image.BILINEAR, expand=False).permute(2, 0, 1) for image, angle in zip(images_list, angles)]
    # rotated_lines_list = [rotate_line(line, angle) for line, angle in zip(lines_list, angles)]
    rotated_images_list, rotated_lines_list = images_list.copy(), lines_list.copy()

    random_indices = generate_random_indices(len(images_list), nbr_to_rotate)
    for i in random_indices:
        # Select initial image and lines
        image = images_list[i]
        line = lines_list[i]
        # Rotate image and line by 90°
        rotated_image = torch.rot90(image)
        rotated_images_list[i] = rotated_image

        rotated_line = ([-line[1][0], -line[1][1]], [line[0][0], line[0][1]])  # ([-y1, -y2], [x1, x2])
        rotated_lines_list[i] = rotated_line

    return rotated_images_list, rotated_lines_list
