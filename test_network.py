from plot.lines_visualisation import create_multiplots
from utils.statistics import mean_square_error, calculate_std_dev, calculate_average_error
from models.model import AngleNet

import copy

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split

from linegeneration.generate_lines import create_image_set
from utils.save_model import save_model
from utils.angle_operations import normalize_angle
from utils.misc import load_list_from_file

import torch

if __name__ == '__main__':

    # Load data

    N = 18
    model = AngleNet(N)
    model_name = 'best_model_1.pt'
    path_model = f"saved\{model_name}"
    model.load_state_dict(torch.load(path_model), strict=False)

    path_tensor = "saved\single_dot_patches_rot.pt"
    tensor_patches = torch.load(path_tensor)

    angles_test_prediction = model(tensor_patches)  # feedforward of the test images
    angles_test_prediction_numpy = angles_test_prediction.detach().numpy()  # convert to numpy array (remove gradient)

    path_angles = "saved\single_dot_normalized_angles_rot.txt"
    angles_lines = load_list_from_file(path_angles)
    # angles_test_prediction_rotated = model(tensor_patches_rotated)
    # angles_test_prediction_numpy_rotated = angles_test_prediction_rotated.detach().numpy()

    # Generate plot
    fig1, axes1 = create_multiplots(tensor_patches, angles_lines, angles_test_prediction_numpy, number_sample=25)
    plt.tight_layout()
    plt.show()

    # fig2, axes2 = create_multiplots(stacked_patches_rotated, angles_lines_rotated, angles_test_prediction_numpy_rotated, number_sample=25)
    # plt.tight_layout()
    # plt.show()

    # Calculate mean square error, standard deviation and average error
    std_dev = calculate_std_dev(angles_lines, angles_test_prediction_numpy)
    # std_dev_rotated = calculate_std_dev(angles_lines_rotated_normalized, angles_test_prediction_numpy_rotated)

    mse = mean_square_error(angles_lines, angles_test_prediction_numpy)
    # mse_rotated = mean_square_error(angles_lines_rotated_normalized, angles_test_prediction_numpy_rotated)

    avg_error = calculate_average_error(angles_lines*np.pi, angles_test_prediction_numpy*np.pi)
    # avg_error_rotated = calculate_average_error(angles_lines_rotated_normalized*pi, angles_test_prediction_numpy_rotated*(2*pi))

    print('MSE regular set: ', "{:.4f}".format(mse))
    # print('MSE rotated set: ', "{:.4f}".format(mse_rotated))

    print('Average error regular set (°): ', "{:.4f}".format(avg_error))
    # print('Average error rotated set (°): ', "{:.4f}".format(avg_error_rotated))

    print('Standard deviation: ', std_dev)
    # print('Standard deviation (rotated set): ', std_dev_rotated)
