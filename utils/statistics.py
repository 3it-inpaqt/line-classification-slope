import torch
import numpy as np
from numpy import ndarray
from collections import Counter

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


def remove_elements_exceeding_count(list1, list2, n):
    """
    Remove elements from list2 for which amount of element in list 1 exceeds n
    :param list1:
    :param list2:
    :param n:
    :return:
    """
    counts = Counter(list1)
    return [element2 for element1, element2 in zip(list1, list2) if counts[element1] <= n]


def resample_dataset(patch_list, angle_list, line_list, threshold):
    """
    Resample dataset to have an even distribution of angles
    :param line_list:
    :param patch_list:
    :param angle_list:
    :param threshold:
    :return: Tuple od resampled patches, angles anf lines
    """
    # Remove values above the threshold

    resampled_angles = remove_elements_exceeding_count(np.around(angle_list, 3), angle_list, threshold)
    resampled_patch = remove_elements_exceeding_count(np.around(angle_list, 3), patch_list, threshold)
    resampled_lines = remove_elements_exceeding_count(np.around(angle_list, 3), line_list, threshold)

    return resampled_patch, resampled_angles, resampled_lines
