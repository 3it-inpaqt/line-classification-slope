import numpy as np
np.seterr(divide='ignore')

def calculate_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the angle of a lign with respect to the
    :param x1: x position of first point
    :param y1: y position of first point
    :param x2: x position of second point
    :param y2: y position of second point
    :return: angle of a line between (x1,y1) and (x2,y2) with respect to the x-axis
    """
    if x1 == x2:
        return np.pi/2
    else:
        slope = (y1 - y2) / (x1 - x2)
        angle = np.arctan(slope)
        return angle


def normalize_angle(angle: float) -> float:
    """
    Normalize angle in radian to a value between 0 and 1
    :param angle: angle of a line
    :return: normalized angle value
    """
    return angle / np.pi
