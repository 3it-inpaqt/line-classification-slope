from numpy import ndarray
import numpy as np


def calculate_angle(image: ndarray) -> tuple[float, float]:
    y, x = np.nonzero(image)                                                        # retrieve the 1 positions
    angle_radian = np.mean(np.arctan2(-(y.mean() - y), x.mean() - x)) + np.pi/2     # calculate angle
    angle_degrees = np.rad2deg(angle_radian)                                        # convert to degree

    return angle_radian, angle_degrees
