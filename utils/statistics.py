import torch
import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
from utils.angle_operations import get_angle_stat


def calculate_std_dev(pred_angles: ndarray, known_angles: ndarray) -> float:
    """
    Calculates the standard deviation between predicted and known angles.
    :param pred_angles: Array of predicted angles
    :param known_angles: Array of known (expected) angles
    :return: Standard deviation between predicted and known angles
    """
    residuals = pred_angles - known_angles
    mean_residuals = np.mean(residuals)
    squared_diff = (residuals - mean_residuals)**2
    variance_residuals = np.mean(squared_diff)
    std_dev_residuals = np.sqrt(variance_residuals)
    return std_dev_residuals


def mean_square_error(observed_value: ndarray, predicted_value:ndarray) -> float:
    """
    Calculate the mean square error, input are expected to be 1D arrays of same length
    :param observed_value:
    :param predicted_value:
    :return: MSE
    """
    mse = np.mean((observed_value - predicted_value) ** 2)
    return mse


def calculate_average_error(actual_values: ndarray, predicted_values: ndarray) -> float:
    """
    Calculate the average error on predictions
    :param actual_values:
    :param predicted_values:
    :return:
    """
    # Calculate absolute differences
    absolute_diffs = np.abs(predicted_values - actual_values)

    # Calculate average error
    average_error = np.mean(absolute_diffs)

    return average_error


def resample_dataset(patch_list, angle_list, line_list):
    """
    Resample dataset to have an even distribution of angles
    :param line_list:
    :param patch_list:
    :param angle_list:
    :return:
    """
    # Adjust the bin size
    bin_size = 0.01

    # Create histogram
    hist, bins = np.histogram(angle_list, bins=int((max(angle_list) - min(angle_list)) / bin_size))

    # Remove values above the threshold
    threshold = 50
    resampled_angles = []
    resampled_patch = []
    resampled_lines = []

    indices = np.where(hist < threshold)

    for i in indices:
        # Get the angles, patch and lines within the current bin
        bin_start, bin_end = bin[i], bin[i + 1]
        in_bin = [(angle, patch, line) for angle, patch, line in zip(angle_list, patch_list, line_list) if bin_start <= angle < bin_end]

        grouped_angles = [in_bin[0][0]]
        grouped_patch = [in_bin[0][1]]
        grouped_lines = [in_bin[0][2]]

        for bag in in_bin[1:]:
            angle = bag[0]
            patch = bag[1]
            line = bag[2]
            if abs(angle - grouped_angles[-1]) <= threshold:
                grouped_angles.append(angle)
                grouped_patch.append(patch)
                grouped_lines.append(line)

        # Add the grouped values to the resampled list
        resampled_angles.extend(grouped_angles)
        resampled_patch.extend(grouped_patch)
        resampled_lines(grouped_lines)

    return resampled_patch, resampled_angles, resampled_lines



