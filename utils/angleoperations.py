from skimage.transform import hough_line
from numpy import ndarray
import numpy as np

from skimage.transform import probabilistic_hough_line
import numpy as np


def calculate_angle(image):
    # Apply probabilistic Hough transform to detect line segments
    lines = probabilistic_hough_line(image, threshold=10, line_length=5, line_gap=3)

    # Find line segment with the longest length
    max_length = 0
    longest_line = None
    for line in lines:
        p0, p1 = line
        length = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        if length > max_length:
            max_length = length
            longest_line = line

    # Calculate angle of longest line segment
    if longest_line is not None:
        p0, p1 = longest_line
        angle = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
        angle = angle % np.pi
        # if angle > np.pi:
        #    angle = np.pi - angle
        return angle
    else:
        return None
