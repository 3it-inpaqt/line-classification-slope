import matplotlib.pyplot as plt
import torch

from models.model import AngleNet
from plot.lines_visualisation import create_multiplots
from utils.misc import load_list_from_file
from utils.settings import settings
# from utils.statistics import mean_square_error, calculate_std_dev, calculate_average_error


if __name__ == '__main__':

    # Load data
    #
    # input_size = settings.patch_size_x * settings.patch_size_y  # image size
    # model = AngleNet(input_size)  # load model template
    # model_type = 'regression'
    #
    # model_name = f'best_model_synthetic_{model_type}_MSE_batch16_epoch1000_2.pt'  # select model to load, make sure you trained one and save it beforehand
    # path_model = f"saved/model/{model_type}/{model_name}"
    # model.load_state_dict(torch.load(path_model), strict=False)  # load the file into the preset model
    #
    # path_tensor = "saved\double_dot_patches_Dx_normalized.pt"  # select tensor file to use, make sure you loaded patches beforehand
    # tensor_patches = torch.reshape(torch.load(path_tensor).squeeze(1), (371, input_size))
    #
    # angles_test_prediction = model(tensor_patches)  # feedforward of the test images
    # angles_test_prediction_numpy = angles_test_prediction.detach().numpy()  # convert to numpy array (remove gradient)
    #
    # path_angles = "saved\double_dot_normalized_angles.txt"  # select angles file to use, make sure you loaded patches beforehand
    # angles_lines = load_list_from_file(path_angles)  # convert file to list
    #
    # # Generate plot
    # for _ in range(1):  # makes several plot, subplots aren't legible with too many image (5x5 is a good size)
    #     fig1, axes1 = create_multiplots(tensor_patches, angles_lines, angles_test_prediction_numpy, number_sample=16)
    #     # plot_patch_test(tensor_patches, sample_number=36, angles_list=angles_lines, predicted_angle=angles_test_prediction_numpy, name='test_DQD')
    #     plt.tight_layout()
    #     plt.show()

    # Calculate mean square error, standard deviation and average error
    # std_dev = calculate_std_dev(angles_lines, angles_test_prediction_numpy)
    # std_dev_rotated = calculate_std_dev(angles_lines_rotated_normalized, angles_test_prediction_numpy_rotated)

    # mse = mean_square_error(angles_lines, angles_test_prediction_numpy)
    # mse_rotated = mean_square_error(angles_lines_rotated_normalized, angles_test_prediction_numpy_rotated)

    # avg_error = calculate_average_error(angles_lines*np.pi, angles_test_prediction_numpy*np.pi)
    # avg_error_rotated = calculate_average_error(angles_lines_rotated_normalized*pi, angles_test_prediction_numpy_rotated*(2*pi))

    # print('MSE regular set: ', "{:.4f}".format(mse))
    # print('MSE rotated set: ', "{:.4f}".format(mse_rotated))

    # print('Average error regular set (°): ', "{:.4f}".format(avg_error))
    # print('Average error rotated set (°): ', "{:.4f}".format(avg_error_rotated))

    # print('Standard deviation: ', std_dev)
    # print('Standard deviation (rotated set): ', std_dev_rotated)

    X = torch.load('test_bastien.pt')
    print(X.shape)
    path_model = '.\\saved\\louis_gaudreau\\regression\\WeightedSmoothL1\\experimental_louis_gaudreau_regression_WeightedSmoothL1005_batch18_epoch200_Dx_1.pt'
    input_size = settings.patch_size_x * settings.patch_size_y  # image size
    model = AngleNet(input_size, 2)  # load model template
    model_type = 'regression'
    model.load_state_dict(torch.load(path_model), strict=False)  # load the file into the preset model
    tensor_patches = torch.reshape(X.squeeze(1), (1, input_size))

    angles_test_prediction = model(tensor_patches)  # feedforward of the test images
    angles_test_prediction_numpy = angles_test_prediction.detach().numpy()  # convert to numpy array (remove gradient)

    print(angles_test_prediction_numpy * 360 - 180)
