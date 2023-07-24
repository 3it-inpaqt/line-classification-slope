"""
Line rotation methods:

This is useful on single dot diagrams as only one orientation of line is possible. The goal is to expend the database
with more angle values. However, it raises the issue of pixel modification, as not all rotation angles are allowed,
only multiples of 90°. It is recommended to work directly on double dots.
"""

import numpy as np
from random import sample
import torch
from typing import Any, List, Tuple

from utils.statistics import angle_distribution
from utils.angle_operations import angle_from_line


def rotate_patches(patch_list, lines_list, angle_list):
    """
    Rotate patches to even out the line orientation distribution
    :param patch_list:
    :param lines_list:
    :param angle_list:
    :return:
    """
    # Find indices of horizontal and vertical lines
    horizontal_indices = [i for i, angle in enumerate(angle_list) if 0 <= angle*360 <= 45 or 45 <= angle <= 180]
    vertical_indices = [i for i, angle in enumerate(angle_list) if 45 <= angle*360 <= 135]

    horizontal_count, vertical_count = len(horizontal_indices), len(vertical_indices)
    target_count = (horizontal_count + vertical_count) // 2

    # print(target_count)
    # print(horizontal_count)
    # print(vertical_count)

    if horizontal_count > vertical_count:
        sampled_indices = sample(horizontal_indices, k=horizontal_count - target_count)
        # print(sampled_indices)
        for index in sampled_indices:
            lines_list[index] = rotate_line_coordinates(lines_list[index], 90)
            # print(lines_list[index])
            angle_list[index] = angle_from_line(lines_list[index])
            patch_list[index] = np.rot90(patch_list[sampled_indices])

    else:
        sampled_indices = sample(vertical_indices, k=vertical_count - target_count)
        # print(sampled_indices)
        for index in sampled_indices:
            lines_list[index] = rotate_line_coordinates(lines_list[index], -90)
            # print(lines_list[index])
            angle_list[index] = angle_from_line(lines_list[index])
            patch_list[index] = np.rot90(patch_list[index])

    return patch_list, lines_list, angle_list


def rotate_line_coordinates(line: List[Tuple[List[Any]]], angle: float):
    """
    Using line coordinates (x1, x2), (y1, y2), rotates a line with a set angle
    :param line:
    :param angle:
    :return:
    """
    # print(line)
    x1, x2 = line[0][0]
    y1, y2 = line[0][1]
    angle_rad = np.radians(angle)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    x1_rot = np.cos(angle_rad) * (x1 - center_x) - np.sin(angle_rad) * (y1 - center_y) + center_x
    y1_rot = np.sin(angle_rad) * (x1 - center_x) + np.cos(angle_rad) * (y1 - center_y) + center_y
    x2_rot = np.cos(angle_rad) * (x2 - center_x) - np.sin(angle_rad) * (y2 - center_y) + center_x
    y2_rot = np.sin(angle_rad) * (x2 - center_x) + np.cos(angle_rad) * (y2 - center_y) + center_y

    return [([int(x1_rot), int(x2_rot)], [int(y1_rot), int(y2_rot)])]

