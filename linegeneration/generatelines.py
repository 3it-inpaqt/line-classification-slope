from typing import Tuple

import numpy as np
from numpy import ndarray
from random import choice
from skimage.draw import line
from utils.angleoperations import calculate_angle


def generate_image(size: tuple, color=None) -> Tuple[ndarray, float]:
    """
    Generate a binary image with a line, which angle is randomly selected
    :param size: size/shape of the image
    :param color: Parameter to decide if the image should have a random background (black or white). By default its value
    is none meaning the background will be randomly chosen between black and white
    :return:
    """
    if color is None:
        background = choice([0, 1])  # randomly set the background to dark or light

    else:
        background = color

    img = np.full(shape=size, fill_value=background, dtype=np.uint8)

    # Select two random positions in the array
    index1 = np.random.choice(img.shape[0], 2, replace=False)
    x1, y1 = tuple(index1)
    # Set a minimum length for the line (at least half the size of the picture)
    length = 0
    while length < size[0] / 2:
        index2 = np.random.choice(img.shape[0], 2, replace=False)
        x2, y2 = tuple(index2)
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Compute angle of the lign with respect to the x-axis (horizontal)
    angle = calculate_angle(x1, y1, x2, y2)

    # Create line starting from (x1,y1) and ending at (x2,y2)
    rr, cc = line(x1, y1, x2, y2)
    img[rr, cc] = 1 - background  # light shade is opposite of the one of the background (1 -> 0 and vice versa)

    return img, angle


def create_image_set(n: int, N: int, color: int = None) -> Tuple[ndarray, ndarray]:
    """
    Generate a batch of arrays with various lines orientation

    :param n: number of image to generate
    :param N: side of each image
    :param color:
    :return: 3d numpy array, n x N x N
    """
    image_set = np.zeros((n, N, N))
    angle_list = []

    for k in range(n):
        image, angle = generate_image((N, N), color=color)
        image_set[k, :, :] = image
        angle_list.append(angle)

    return image_set, np.array(angle_list)
