from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error
import torch
from typing import Any

from utils.logger import logger
# from utils.angle_operations import get_angle_stat


# -- Statistics variables -- #

def calculate_std_dev(pred_angles: Any, known_angles: Any) -> float:
    """
    Calculates the standard deviation between predicted and known angles.
    :param pred_angles: Array of predicted angles
    :param known_angles: Array of known (expected) angles
    :return: Standard deviation between predicted and known angles
    """
    # Convert input to NumPy arrays if they are PyTorch tensors
    if torch.cuda.is_available():
        if isinstance(pred_angles, torch.Tensor):
            pred_angles = pred_angles.cpu().detach().numpy()
        if isinstance(known_angles, torch.Tensor):
            known_angles = known_angles.cpu().detach().numpy()
    else:
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


def accuracy(known_angles, pred_angles, tol=0.1):
    # Convert input to NumPy arrays if they are PyTorch tensors
    if torch.cuda.is_available():
        if isinstance(pred_angles, torch.Tensor):
            pred_angles = pred_angles.cpu().detach().numpy()
        if isinstance(known_angles, torch.Tensor):
            known_angles = known_angles.cpu().detach().numpy()
    else:
        if isinstance(pred_angles, torch.Tensor):
            pred_angles = pred_angles.detach().numpy()
        if isinstance(known_angles, torch.Tensor):
            known_angles = known_angles.detach().numpy()

    return 1 - mean_absolute_error(known_angles, pred_angles)


def mean_square_error(observed_value: ndarray, predicted_value:ndarray) -> ndarray:
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


# -- Resampling operations -- #

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

    resampled_angles = remove_elements_exceeding_count(np.around(angle_list, 2), angle_list, threshold)
    resampled_patch = remove_elements_exceeding_count(np.around(angle_list, 2), patch_list, threshold)
    resampled_lines = remove_elements_exceeding_count(np.around(angle_list, 2), line_list, threshold)

    return resampled_patch, resampled_angles, resampled_lines


# -- Study relations between standard deviation/loss and settings
def plot_metrics():
    """
    Create plots for each setting to characterize relationships between accuracy metrics
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })
    # Define the folder name where the CSV files are saved
    folder = './saved/csv_files'

    # Get the list of CSV files in the folder
    csv_files = [file for file in os.listdir(folder) if file.endswith('.csv')]

    # Iterate over the CSV files and plot the data
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(folder, csv_file))
        column_names = df.columns.tolist()[:-2]
        num_cols = df.shape[1] - 2
        # print(column_names)

        # Get the setting name from the file name
        setting_name = os.path.splitext(csv_file)[0]
        # print(setting_name)

        # Create subplots for each setting
        fig, axes = plt.subplots(num_cols, 2, figsize=(num_cols*2, num_cols*2))
        fig.suptitle(setting_name, fontsize=28)

        # Plot the standard deviation
        for i in range(num_cols):
            df_sorted = df.sort_values(by=column_names[i])

            # Group the data by the fixed parameters
            fixed_params = [col for col in column_names if col != column_names[i]]
            grouped_data = df_sorted.groupby(fixed_params)

            for group_name, group_data in grouped_data:
                ax_left = axes[i, 0]
                ax_left.plot(group_data[column_names[i]], group_data['Standard Deviation'], label=str(group_name))
                ax_left.set_xlabel(column_names[i], fontsize=16)
                ax_left.set_ylabel('STD Deviation', fontsize=16)

                ax_right = axes[i, 1]
                ax_right.plot(group_data[column_names[i]], group_data['Loss'], label=str(group_name))
                ax_right.set_xlabel(column_names[i], fontsize=16)
                ax_right.set_ylabel('Loss', fontsize=16)

            # Add the legend with the specified properties
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axes[i, 0].legend(title='Fixed Params', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
                              prop=props)
            axes[i, 1].legend(title='Fixed Params', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
                              prop=props)

    # Adjust the spacing between subplots
    plt.tight_layout()
    # Save the figure
    # plt.savefig('./saved/plots/accuracy_metrics.png')
    # Show the plot
    plt.show()

    logger.info('Plots created for accuracy metrics')


