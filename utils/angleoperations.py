import numpy as np


def calculate_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the angle of a lign with respect to the
    :param x1: x position of first point
    :param y1: y position of first point
    :param x2: x position of second point
    :param y2: y position of second point
    :return:
    """
    try:
        slope = (y1 - y2) / (x1 - x2)
        angle = np.arctan(slope)
        return angle

    except ZeroDivisionError:
        return np.pi/2
