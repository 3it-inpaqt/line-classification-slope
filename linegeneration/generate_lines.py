import numpy as np
from numpy import ndarray
from skimage.draw import line, line_aa
from scipy.ndimage import gaussian_filter  # only import if necessary
from typing import Tuple

from utils.angle_operations import calculate_angle, normalize_angle


def generate_image(size: tuple, gaussian_blur: bool = False) -> Tuple[ndarray, float]:
    """
    Generate a binary image with a random line
    :param size: Shape of the image
    :param gaussian_blur: Add a gaussian blur to the image if True
    :return:
    """
    img = np.random.normal(10, 2, size) * 255
    min_length = 0.5 * size[0]

    # Select two random positions in the array
    index1 = np.random.choice(img.shape[0], 2, replace=False)
    x1, y1 = tuple(index1)
    # Set a minimum length for the line (at least half the size of the picture)
    length = 0
    while length <= min_length:  # while the length is not at least half the size of the picture it selects new endpoints
        index2 = np.random.choice(img.shape[0], 2, replace=False)
        x2, y2 = tuple(index2)
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Compute angle of the line with respect to the x-axis (horizontal)
    angle = calculate_angle(x1, y1, x2, y2)

    # Create line starting from (x1,y1) and ending at (x2,y2)
    rr, cc = line(x1, y1, x2, y2)
    img[rr, cc] = 255

    if gaussian_blur:
        img = gaussian_filter(img, sigma=0.8)

    return img/255, normalize_angle(angle)


def create_image_set(n: int, N: int, gaussian_blur: bool = False) -> Tuple[ndarray, ndarray]:
    """
    Generate a batch of arrays with various lines orientation

    :param n: number of image to generate
    :param N: side of each image
    :param gaussian_blur: Add a gaussian blur to the image if True
    :return: 3d numpy array, n x N x N
    """
    image_set = np.zeros((n, N, N))  # important for NN to have size n x N x N
    angle_list = []

    for k in range(n):
        image, angle = generate_image((N, N), gaussian_blur)
        image_set[k, :, :] = image
        angle_list.append(angle)

    return image_set, np.array(angle_list)
