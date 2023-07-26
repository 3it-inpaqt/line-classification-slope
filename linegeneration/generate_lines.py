import numpy as np
from numpy import ndarray
from random import uniform, random
from skimage.draw import line, line_aa
from scipy.ndimage import gaussian_filter  # only import if necessary
from typing import Tuple

from utils.angle_operations import calculate_angle, normalize_angle

import math


def generate_image(size: tuple, background: bool = False, sigma: float = 0, aa: bool = False, generation_type: str = 'random') -> Tuple[ndarray, float]:
    """
    Generate a binary image with a random line
    :param size: Shape of the image
    :param background: Whether to make a noisy background or not
    :param sigma: Add a gaussian blur to the image if True
    :param aa: Anti-alias, creates AA line or not
    :param generation_type: Set how the lines are generated (random, vertical, horizontal, fix)
    :return: A single patch with the angle's value of its line
    """
    # Generate gaussian distribution for background if specified
    if background:
        img = np.random.normal(0.1, 0.01, size) * 255
    # Otherwise blank background (black)
    else:
        img = np.zeros(size)

    min_length = 0.9 * min(size[0], size[1])

    # TODO Fix this because the vertical and horizontal settings won't do anything
    if generation_type == 'random':
        # Select one random position on an edge
        index1 = np.random.choice([0, size[0] - 1]), np.random.choice(size[1])
        x1, y1 = tuple(index1)

        # Select another random position on a different edge
        index2 = np.random.choice([0, size[1] - 1]), np.random.choice(size[1])
        x2, y2 = tuple(index2)

        # Set a minimum length for the line (at least half the size of the picture)
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        while length <= min_length:  # while the length is not at least half the size of the picture it selects new endpoints
            index2 = np.random.choice([0, size[1] - 1]), np.random.choice(size[1])
            x2, y2 = tuple(index2)
            length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # Compute angle of the line with respect to the x-axis (horizontal)
        angle = calculate_angle(x1, y1, x2, y2)

        # Create line starting from (x1,y1) and ending at (x2,y2)
        if aa:
            rr, cc, val = line_aa(x1, y1, x2, y2)  # thicc line if anti-alias is on
            img[rr, cc] = 255 * val
            # print(val)
        else:
            rr, cc = line(x1, y1, x2, y2)  # one pixel thicc line otherwise
            img[rr, cc] = 255

    else:
        if generation_type == 'vertical':
            angle = uniform(45, 135)

        elif generation_type == 'horizontal':
            # Generate a random float within each range
            angle_1 = uniform(0, 45)
            angle_2 = uniform(135, 180)

            # Randomly select one of the two generated values
            if random() < 0.5:
                angle = angle_1
            else:
                angle = angle_2

        else:
            raise 'Illegal generation type, please select random, vertical or horizontal'

    # Add a gaussian blur if specified
    if sigma > 0:
        img = gaussian_filter(img, sigma=sigma)

    return img/255, normalize_angle(angle)


def create_image_set(n: int, N: int, background: bool = False, gaussian_blur: float = 0, aa: bool = False, generation_type: str = 'random') -> Tuple[ndarray, ndarray]:
    """
    Generate a batch of arrays with various lines orientation

    :param n: number of image to generate
    :param N: side of each image
    :param background: Whether to make a noisy background or not
    :param gaussian_blur: Add a gaussian blur to the image if True
    :param aa: Anti-alias, creates AA line or not
    :param generation_type: Set how the lines are generated (random, vertical, horizontal)
    :return: A tuple of a 3d numpy array, n x N x N and a list of angles for each patch contained in the array
    """
    # Initialise outputs
    image_set = np.zeros((n, N, N))  # important for NN to have size n x N x N
    angle_list = []

    # Iterates to generate n images and put them in the output array
    for k in range(n):
        image, angle = generate_image((N, N), background=background, sigma=gaussian_blur, aa=aa, generation_type=generation_type)
        image_set[k, :, :] = image
        angle_list.append(angle)

    return image_set, np.array(angle_list)
