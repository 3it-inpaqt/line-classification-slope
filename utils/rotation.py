import numpy as np
from random import sample
from typing import Any, List, Tuple

from utils.angle_operations import angle_from_line


def rotate_patches(patch_list: List[Any],
                   lines_list: List[Any],
                   angle_list: List[Any]) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Rotate patches to even out the line orientation distribution
    :param patch_list: List of experimental patches
    :param lines_list: List of the lines endpoints coordinates
    :param angle_list: List of the lines angle of each patch
    :return: A new tuple with patches, lines coordinates and angles rotated
    """
    # Find indices of horizontal and vertical lines
    horizontal_indices = [i for i, angle in enumerate(angle_list) if 0 <= angle*360 <= 45 or 135 <= angle*360 <= 180]
    vertical_indices = [i for i, angle in enumerate(angle_list) if 45 <= angle*360 <= 135]

    # Count the number of horizontal and vertical line
    horizontal_count, vertical_count = len(horizontal_indices), len(vertical_indices)
    # Set a target number for the distribution (here 50/50)
    target_count = max(horizontal_count, vertical_count) // 2

    # Creates copy of the input to avoid issues
    rotated_patch_list = patch_list.copy()
    rotated_lines_list = lines_list.copy()
    rotated_angle_list = angle_list.copy()

    # If there are more horizontal lines, select some to even out the dataset
    if horizontal_count > vertical_count:
        # Select random indices in the list of horizontal lines
        sampled_indices = sample(horizontal_indices, k=target_count)
        for index in sampled_indices:
            # Rotate the lines coordinates
            rotated_lines_list[index] = rotate_line_coordinates(lines_list[index], 90)
            # Calculate the new angle to ensure it's the correct one (doesn't really matter)
            rotated_angle_list[index] = angle_from_line(rotated_lines_list[index], normalize=True)
            # Rotate the patch
            rotated_patch_list[index] = np.rot90(patch_list[index])

    # If there are more vertical lines, select some to even out the dataset
    else:
        # Select random indices in the list of vertical lines
        sampled_indices = sample(vertical_indices, k=target_count)
        for index in sampled_indices:
            # Rotate the lines coordinates
            rotated_lines_list[index] = rotate_line_coordinates(lines_list[index], 90)
            # Calculate the new angle to ensure it's the correct one (doesn't really matter)
            rotated_angle_list[index] = angle_from_line(rotated_lines_list[index], normalize=True)
            # Rotate the patch
            rotated_patch_list[index] = np.rot90(patch_list[index])

    return rotated_patch_list, rotated_lines_list, rotated_angle_list


def rotate_line_coordinates(line: List[Tuple[List[Any]]], angle: float = 90) -> List[Any]:
    """
    Using line coordinates (x1, x2), (y1, y2), rotates a line with a set angle
    :param line: Line endpoint coordinates [([x1, x2], [y1, y2])]
    :param angle: Angle of rotation, in degrees
    :return: New line coordinates [([x1', x2'], [y1', y2'])]
    """
    # Fetch line coordinates
    x1, x2 = line[0][0]
    y1, y2 = line[0][1]
    # Convert angle to radian
    angle_rad = np.radians(angle)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Apply rotation on coordinates
    x1_rot = np.cos(angle_rad) * (x1 - center_x) - np.sin(angle_rad) * (y1 - center_y) + center_x
    y1_rot = np.sin(angle_rad) * (x1 - center_x) + np.cos(angle_rad) * (y1 - center_y) + center_y
    x2_rot = np.cos(angle_rad) * (x2 - center_x) - np.sin(angle_rad) * (y2 - center_y) + center_x
    y2_rot = np.sin(angle_rad) * (x2 - center_x) + np.cos(angle_rad) * (y2 - center_y) + center_y

    # Makes sure the output is a set of integers
    return [([int(x1_rot), int(x2_rot)], [int(y1_rot), int(y2_rot)])]


