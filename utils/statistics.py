import numpy as np
import torch
from typing import Any


def calculate_std_dev(pred_angles: Any, known_angles: Any) -> float:
    """
    Calculates the standard deviation between predicted and known angles.
    :param pred_angles: Array of predicted angles
    :param known_angles: Array of known (expected) angles
    :return: Standard deviation between predicted and known angles
    """
    # Convert input to NumPy arrays if they are PyTorch tensors
    if isinstance(pred_angles, torch.Tensor):
        pred_angles = pred_angles.detach().numpy()
    if isinstance(known_angles, torch.Tensor):
        known_angles = known_angles.detach().numpy()

    # Calculate standard deviation
    residuals = pred_angles - known_angles
    mean_residuals = np.mean(residuals)
    variance_residuals = np.sum((residuals - mean_residuals) ** 2) / (len(residuals) - 1)
    std_dev_residuals = np.sqrt(variance_residuals)
    return std_dev_residuals
