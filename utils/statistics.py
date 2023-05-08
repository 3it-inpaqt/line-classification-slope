import torch
import numpy as np
from numpy import ndarray


def calculate_std_dev(pred_angles: ndarray, known_angles: ndarray) -> float:
    """
    Calculates the standard deviation between predicted and known angles.
    :param pred_angles: Array of predicted angles
    :param known_angles: Array of known (expected) angles
    :return: Standard deviation between predicted and known angles
    """
    residuals = pred_angles - known_angles
    mean_residuals = np.mean(residuals)
    variance_residuals = np.sum((residuals - mean_residuals) ** 2) / (len(residuals) - 1)
    std_dev_residuals = np.sqrt(variance_residuals)
    return std_dev_residuals

