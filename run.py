from linegeneration.generatelines import create_image_set
from plot.linesvizualisation import create_multiplots
import matplotlib.pyplot as plt
import torch

from models.model import AngleNet
from utils.statistics import calculate_std_dev
from utils.angleoperations import normalize_angle

if __name__ == '__main__':
    # Generates images
    N = 18
    n = 16

    # Load model
    model = AngleNet(N)
    model_name = 'best_model_synthetic_LeakyReLU_shade.pt'
    path = f"saved\model\{model_name}"
    model.load_state_dict(torch.load(path), strict=False)

    # Apply model
    image_set_test, angles_test = create_image_set(n, N)  # generate new image set to test the network on new images
    angles_test_normalized = normalize_angle(angles_test)
    tensor_image_test = torch.tensor(image_set_test, dtype=torch.float32)  # convert ndarray to tensor and flatten it
    tensor_image_test = tensor_image_test.flatten(1)
    angles_test_prediction = model(tensor_image_test)  # feedforward of the test images
    angles_test_prediction_numpy = angles_test_prediction.detach().numpy()

    # Generate plot
    fig, axes = create_multiplots(image_set_test, angles_test, angles_test_prediction_numpy)

    # patches, angles = create_image_set(n, N)
    #
    # fig, axes = create_multiplots(patches, angles)
    plt.tight_layout()
    plt.show()

    # # Calculate standard deviation
    # std_dev = calculate_std_dev(angles_test_normalized, angles_test_prediction_numpy)
    #
    # print('Standard deviation: ', std_dev)
