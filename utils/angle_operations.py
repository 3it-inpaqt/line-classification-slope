import numpy as np
from numpy import ndarray
import random

import torch
import torchvision.transforms.functional as TF

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


def rotate_line(line_coords, index, angle):
    """
    Rotate a line using its coordinates
    :param line_coords: Tuple containing coordinates lists
    :param index: What index to select to rotate the line
    :param angle: Angle of rotation
    :return:
    """
    # Select the line coordinates using the provided index
    line = line_coords[index]

    # Extract the x and y coordinates of the line
    x_coords, y_coords = line

    # Convert the coordinates to torch tensors
    x_tensor = torch.tensor(x_coords)
    y_tensor = torch.tensor(y_coords)

    # Calculate the center point of the line
    center = torch.tensor([(x_tensor[0] + x_tensor[1]) / 2, (y_tensor[0] + y_tensor[1]) / 2])

    # Define the rotation matrix based on the provided angle
    rotation_matrix = torch.tensor([[torch.cos(torch.deg2rad(angle)), -torch.sin(torch.deg2rad(angle))],
                                    [torch.sin(torch.deg2rad(angle)), torch.cos(torch.deg2rad(angle))]])

    # Apply the rotation to the line coordinates
    rotated_coords = torch.matmul(rotation_matrix, torch.stack([x_tensor - center[0], y_tensor - center[1]]))
    rotated_coords = rotated_coords + center.unsqueeze(1)

    # Convert the rotated coordinates back to lists
    rotated_x_coords = rotated_coords[0].tolist()
    rotated_y_coords = rotated_coords[1].tolist()

    # Update the line coordinates with the rotated coordinates
    line_coords[index] = (rotated_x_coords, rotated_y_coords)

    # Return the updated line coordinates
    return line_coords


def random_rotate_images(images_list, lines_list, angle_range=(-90, 90)):
    """
    Rotate images randomly with corresponding line
    :param images_list:
    :param lines_list:
    :param angle_range:
    :return:
    """
    # Randomly select an index
    index = random.randint(0, len(images_list) - 1)

    # Select the image tensor and associated line coordinates
    image_tensor = images_list[index]
    line_coords = lines_list[index]

    # Generate a random angle within the specified range
    angle = random.uniform(angle_range[0], angle_range[1])

    # Rotate the image tensor
    rotated_tensor = TF.rotate(image_tensor, angle)

    # Rotate the line coordinates
    rotated_line_coords = rotate_line(line_coords, index, angle)

    # Return the rotated image tensor and updated line coordinates
    return rotated_tensor, rotated_line_coords
