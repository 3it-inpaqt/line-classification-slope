from skimage.transform import hough_line
from numpy import pi, ndarray


def calculate_angle(img: ndarray) -> tuple[float, float]:
    # Apply Hough transform to get the most prominent line
    _, angles, d = hough_line(img)
    # Convert angle to degrees
    angle_deg = angles[0] * 180 / pi

    # Handle angles in the third and fourth quadrants
    if angle_deg > 90:
        angle_deg -= 180

    # Return angle in radians and degrees
    return angles[0], angle_deg
