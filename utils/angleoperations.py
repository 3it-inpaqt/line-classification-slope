from numpy import ndarray
import numpy as np


def calculate_angle(image: ndarray) -> tuple[float, float]:
    y, x = np.nonzero(image)                                                        # retrieve the 1 positions
    angle_radian = np.mean(np.arctan2(-(y.mean() - y), x.mean() - x)) + np.pi/2     # calculate angle

    return angle_radian
