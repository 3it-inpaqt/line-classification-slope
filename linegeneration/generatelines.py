import numpy as np
from numpy import ndarray
from skimage.draw import line
from utils.angleoperations import calculate_angle


def generate_image(size: tuple) -> ndarray:
    img = np.zeros(size, dtype=np.uint8)

    # Select two random positions in the array
    index1 = np.random.choice(img.shape[0], 2, replace=False)
    x1, y1 = tuple(index1)
    # Set a minimum length for the line (at least half the size of the picture)
    length = 0
    while length < size[0] / 2:
        index2 = np.random.choice(img.shape[0], 2, replace=False)
        x2, y2 = tuple(index2)
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Create line starting from (x1,y1) and ending at (x2,y2)
    rr, cc = line(x1, y1, x2, y2)
    img[rr, cc] = 1

    return img


def create_batch(n: int, size: tuple[int, int]) -> ndarray:
    """
    Generate a batch of arrays with various lines orientation

    :param n: number of image to generate
    :param size: size of each image
    :return: 3d numpy array, (size) x n
    """
    size_batch = size + (n,)  # concatenate tuple for correct size definition in the array
    batch = np.zeros(size_batch)

    angle_list = []

    for k in range(n):
        image = generate_image(size)
        batch[:, :, k] = image
        angle_radian = calculate_angle(image)
        angle_list.append(angle_radian)

    return batch, angle_list
