import cv2
import numpy as np
import torch

from utils.misc import load_list_from_file
from utils.settings import settings


def main():

    # Load patches
    X_path = settings.x_path
    y_path = settings.y_path
    X, y = torch.load(X_path), [float(x) for x in load_list_from_file(y_path)]

    img = X[100, :].reshape(settings.patch_size_x, settings.patch_size_y).numpy()

    # Check the data type of the input image
    if img.dtype != np.uint8:
        # Convert to 8-bit unsigned integer
        img = (img * 255).astype(np.uint8)

    # Check the number of channels in the input image
    if img.ndim == 3 and img.shape[-1] > 1:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)

    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    # Check if lines is not None before iterating over it
    if lines is not None:
        # Draw lines on the original image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Create a named window and resize it
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 200, 200)

    # Display the image with detected lines
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
