import numpy as np
from numpy import ndarray
import random

import torch
import torchvision.transforms.functional as F

from utils.misc import random_select_elements

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
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    # Apply rotation to line coordinates
    rotated_line = np.dot(rotation_matrix, np.array([[x1, x2], [y1, y2]]))
    # Extract rotated line coordinates
    rotated_x1, rotated_x2 = rotated_line[0]
    rotated_y1, rotated_y2 = rotated_line[1]

    return [rotated_x1, rotated_x2], [rotated_y1, rotated_y2]


def rotate_image(image, angle):
    """
    Rotate image tensor by an angle given by the user. Make sure the angle is in degrees.
    :param image:
    :param angle:
    :return:
    """
    # Get image dimensions
    height, width = image.shape[-2:]
    angle_rad = np.deg2rad(angle)

    # Create a grid of coordinates for each pixel in the image
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width))
    grid = torch.stack((grid_x, grid_y), dim=-1).float()

    # Convert the grid to the range [-1, 1]
    grid_normalized = (grid * 2 / torch.tensor([width - 1, height - 1])) - 1

    # Apply rotation to the grid coordinates
    rotation_matrix = torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad)],
                                    [torch.sin(angle_rad), torch.cos(angle_rad)]])
    rotated_grid = torch.matmul(grid_normalized, rotation_matrix)

    # Convert the rotated grid back to the range [0, width-1] and [0, height-1]
    rotated_grid = ((rotated_grid + 1) / 2) * torch.tensor([width - 1, height - 1])

    # Apply the inverse transformation to the image
    rotated_image = F.grid_sample(image.unsqueeze(0).unsqueeze(0), rotated_grid.unsqueeze(0))

    return rotated_image.squeeze()


def random_rotate_images(images_list, lines_list):
    """
    Rotate images randomly with corresponding line
    :param images_list:
    :param lines_list:
    :return:
    """
    # Randomly select the image tensor and associated line coordinates
    image_tensor, line_coords, indices = random_select_elements(images_list, lines_list)
    # print('Image shape: ', image_tensor.shape)

    # Rotate the tensor assuming image_tensor is a Torch tensor with shape [H, W]
    rotated_tensor_image = torch.transpose(image_tensor, 0, 1).flip(1)
    # Rotate the line coordinates
    rotated_line_coords = rotate_line(line_coords, 90)

    # Update the rotated image tensor and line coordinates
    rotated_tensors = images_list.copy()
    rotated_lines = lines_list.copy()

    rotated_tensors[indices] = rotated_tensor_image
    rotated_lines[indices] = rotated_line_coords

    return rotated_tensors, rotated_lines
