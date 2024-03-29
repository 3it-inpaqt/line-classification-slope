import math
import numpy as np
from skimage.draw import line, line_aa
from scipy.ndimage import gaussian_filter
from typing import Any, List, Tuple

from utils.angle_operations import normalize_angle
import utils.rotation as rotation
from utils.settings import settings


def generate_image_fixed_angle(size: tuple,
                               background: bool = False,
                               sigma: float = 0,
                               aa: bool = False,
                               angle: float = 0) -> Tuple[Any, Any, List[Any]]:
    """
    Generate a binary image with a line of a specified angle (copy of the generate_image function in
    utils.generate_lines.py)
    :param size: Shape of the image
    :param background: Whether to make a noisy background or not
    :param sigma: Add a Gaussian blur to the image if True
    :param aa: Anti-alias, creates AA line or not
    :param angle: Desired angle of the line
    :return: A single patch with the angle's value of its line
    """
    # Generate gaussian distribution for background if specified
    if background:
        img = np.random.normal(settings.mean_gaussian, settings.scale_gaussian, size) * 255
    # Otherwise blank background (black)
    else:
        img = np.zeros(size)

    # Set a minimum line length
    min_length = settings.minimum_length * min(size[0], size[1])

    # Calculate the endpoints of the line based on the desired angle
        # Convert angle to radian
    angle_rad = math.radians(angle)
        # Get line center
    center_x = size[0] / 2
    center_y = size[1] / 2
        # When drawing from center, need to set the radius to total line length divided by 2
    length = min_length / 2
        # Generate line coordinates
    x1 = int(center_x + length * math.cos(angle_rad))
    y1 = int(center_y + length * math.sin(angle_rad))
    x2 = int(center_x - length * math.cos(angle_rad))
    y2 = int(center_y - length * math.sin(angle_rad))

    # Create line starting from (x1,y1) and ending at (x2,y2)
    if aa:
        # Generate line with anti-alias
        rr, cc, val = line_aa(x1, y1, x2, y2)
        img[rr, cc] = 255 * val
    else:
        # Generate a 1-pixel thicc line
        rr, cc = line(x1, y1, x2, y2)
        img[rr, cc] = 255

    # Add a Gaussian blur if specified
    if sigma > 0:
        img = gaussian_filter(img, sigma=sigma)

    img = np.rot90(img)  # makes sure the synthetic and experimental data have the same origin when checking angles

    return img / 255, normalize_angle(angle_rad), [([x1, x2], [y1, y2])]


def populate_angles(patch_list,
                    lines_list, angle_list,
                    percentage: float,
                    size: tuple,
                    background: bool = False,
                    sigma: float = 0,
                    aa: bool = False) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    This function will do two things:
        - first, it will apply rotation to populate the dataset with perpendicular angles
        - second, it will check the range where angles are missing and populate it with synthetic data
    :param patch_list:
    :param lines_list:
    :param angle_list:
    :param percentage: Percentage of synthetic data to generate with respect to the total number of experimental data
    :param size: Shape of the image
    :param background: Whether to make a noisy background or not
    :param sigma: Add a Gaussian blur to the image if True
    :param aa: Anti-alias, creates AA line or not
    :return:
    """
    # Takes the dataset and apply rotation to create two uniform boundaries
    rotated_patches, rotated_lines_list, rotated_angle_list = rotation.rotate_patches(patch_list, lines_list,
                                                                              angle_list)

    # Initialize output
    new_patch_list, new_lines_list, new_angle_list = rotated_patches.copy(), rotated_lines_list.copy(), rotated_angle_list.copy()

    # Find the boundaries to populate missing angles
        # First sort the list of angles
    sorted_angles_list = rotated_angle_list.copy()
    sorted_angles_list.sort()
        # Calculate consecutive difference between values
    differences = np.diff(sorted_angles_list)
        # Find the index where the maximum difference occurs
    max_diff_index = np.argmax(differences)
        # Split list into two subsets based on the index of maximum difference to get boundaries for missing values
    bottom_max = sorted_angles_list[max_diff_index]
    top_min = sorted_angles_list[max_diff_index + 1]

    # Find distribution of rotated patches
    num_horizontal = len([i for i, angle in enumerate(sorted_angles_list) if 0 <= angle * 360 <= 45 or 135 <= angle * 360 <= 180])
    num_vertical = len([i for i, angle in enumerate(sorted_angles_list) if 45 <= angle * 360 <= 135])

    num_to_generate = int((num_horizontal + num_vertical) * percentage)  # set number of synthetic patches to generate
    # Generate missing angles
        # Set randomly generated angles within the ranges defined by bottom_max and top_min
    missing_angles = np.random.uniform(low=bottom_max, high=top_min, size=num_to_generate)
        # Iterates through the new angles to add the synthetic patches to the dataset
    for angle in missing_angles:
        new_patch, new_angle, new_line = generate_image_fixed_angle(size=size, background=background, sigma=sigma, aa=aa, angle=angle*360)
        new_patch_list.append(new_patch)
        new_angle_list = np.append(arr=new_angle_list, values=new_angle)
        new_lines_list.append(new_line)

    return new_patch_list, new_lines_list, new_angle_list
