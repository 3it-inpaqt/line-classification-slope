import numpy as np
from numpy import ndarray
from skimage.draw import line



def generate_image(size: tuple) -> ndarray:
    img = np.zeros(size, dtype=np.uint8)

    # Select two random positions in the array
    index1 = np.random.choice(img.shape[0], 2, replace=False)
    index2 = np.random.choice(img.shape[0], 2, replace=False)
    x1, y1 = tuple(index1)
    x2, y2 = tuple(index2)

    # Create line starting from (x1,y1) and ending at (x2,y2)
    rr, cc = line(x1, y1, x2, y2)
    img[rr, cc] = 1

    return img
