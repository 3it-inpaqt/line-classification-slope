from linegeneration.generate_lines import create_image_set
from plot.lines_visualisation import create_multiplots
import matplotlib.pyplot as plt
import torch

from models.model import AngleNet
from utils.statistics import calculate_std_dev
from utils.angle_operations import normalize_angle

if __name__ == '__main__':
    # Generates images
    N = 18
    n = 16

    # if torch.cuda.is_available():
    #     print('it works')

    # Load model
    # model = AngleNet(N)
    # model_name = 'best_model_LeakyReLU.pt'
    # path = f"saved\model\{model_name}"
    # model.load_state_dict(torch.load(path), strict=False)
    #
    # # Apply model
    # image_set_test, angles_test = create_image_set(n, N)  # generate new image set to test the network on new images
    # angles_test_normalized = normalize_angle(angles_test)  # normalize angle
    # tensor_image_test = torch.tensor(image_set_test, dtype=torch.float32)  # convert ndarray to tensor and flatten it
    # tensor_image_test = tensor_image_test.flatten(1)
    # angles_test_prediction = model(tensor_image_test)  # feedforward of the test images
    # angles_test_prediction_numpy = angles_test_prediction.detach().numpy()  # convert result to numpy array

    # Generate plot
    # fig, axes = create_multiplots(image_set_test, angles_test, angles_test_prediction_numpy)
    # plt.tight_layout()
    # plt.show()
    #
    # # Calculate standard deviation
    # std_dev = calculate_std_dev(angles_test_normalized, angles_test_prediction_numpy)
    #
    # print('Standard deviation: ', std_dev)

    # import numpy as np
    # img = np.random.normal(0.5, 2, (N, N))
    #
    image_set_test, angles_test = create_image_set(n, N, 0.9, aa=True)
    fig, axes = create_multiplots(image_set_test, angles_test, number_sample=n)
    plt.show()

