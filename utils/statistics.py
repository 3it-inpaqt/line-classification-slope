import torch
import numpy as np
from numpy import ndarray


def calculate_std_dev(expected_output: ndarray, predicted_output: ndarray) -> float:
    """
    Calculate the standard deviation between expected outputs and predicted ones
    :param expected_output:
    :param predicted_output:
    :return: standard deviation of the data set
    """
    # Calculate the mean
    mean = np.mean(expected_output)

    # Calculate the standard deviation
    std_dev = np.sqrt(np.sum((predicted_output - expected_output)**2) / (len(expected_output)))

    return std_dev
