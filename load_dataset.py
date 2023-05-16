from typing import List
import torch

import matplotlib.pyplot as plt

from classes.diagram_offline import DiagramOffline
from plot.lines_visualisation import create_multiplots
from utils.statistics import calculate_std_dev
from utils.angle_operations import angles_from_list, normalize_angle, random_rotate_images
from utils.settings import settings
from utils.output import init_out_directory, ExistingRunName
from utils.logger import logger
from pathlib import Path
from classes.qdsd import DATA_DIR
from plot.data import plot_patch_sample
from models.model import AngleNet

run_name = settings.run_name

try:
    # Create the output directory to save results and plots
    init_out_directory()
except ExistingRunName:
    logger.critical(f'Existing run directory: "{run_name}"', exc_info=True)


def load_diagram() -> List["DiagramOffline"]:
    # Load diagrams from files (line and area)
    diagrams = DiagramOffline.load_diagrams(pixel_size=settings.pixel_size,
                                            research_group=settings.research_group,
                                            diagrams_path=Path(DATA_DIR, 'interpolated_csv.zip'),
                                            labels_path=Path(DATA_DIR, 'labels.json'),
                                            single_dot=True,
                                            load_lines=True,
                                            load_areas=True,
                                            white_list=[settings.test_diagram] if settings.test_diagram else None)

    # Normalize the diagram with the same min/max value used during the training.
    # The values are fetch via the "normalization_values_path" setting or in the current run directory.
    # DiagramOffline.normalize_diagrams(diagrams)

    return diagrams


def load_patches(diagrams):
    """
    From a diagrams list, generate patches on each diagram with associated line intersecting them. In case there are more
    than one line cutting the patch, it is stored in a separated list called patches_multi_line. The patch is repeated N times
    corresponding to the N lines intersecting it. It acts like a single line patch, but it might be more clever to differentiate
    these two cases. It might not be the smartest way to do it though.
    :param diagrams:
    :return:
    """
    # Patches with one line
    patches = []
    lines = []

    for diagram in diagrams:
        diagram_patches, lines_patches = diagram.get_patches((settings.patch_size_x, settings.patch_size_y), (6, 6),
                                                             (0, 0))
        patches.extend(diagram_patches)
        lines.extend(lines_patches)

    return patches, lines


if __name__ == '__main__':
    diagrams_exp = load_diagram()
    # print('diagram ', len(diagrams_exp))
    patches_list, lines_list = load_patches(diagrams_exp)
    # print('lines list:', lines_list[:10])
    # plot_patch_sample(patches_list, lines_list, sample_number=25, show_offset=False)

    # Rotate some patches randomly
    rotated_patches, rotated_lines = patches_list.copy(), lines_list.copy()
    number_rotated = len(rotated_lines)//2
    for q in range(number_rotated):
        print(f'iteration #{q}')
        rotated_patches, rotated_lines = random_rotate_images(rotated_patches, rotated_lines)
        print('---------------------------')
    # Calculate angles by hand for verification
    # angles_lines = angles_from_list(lines_list)
    # angles_lines_normalized = normalize_angle(angles_lines)
    #
    # angles_lines_rotated = angles_from_list(rotated_lines)
    # angles_lines_rotated_normalized = normalize_angle(angles_lines_rotated)

    # Reshape patches for neural network
    # Get the number of images and the size of each image
    # n = len(patches_list)
    # N = patches_list[0].shape[0]
    #
    # # Create an empty tensor with the desired shape
    # stacked_patches = torch.empty(n, N, N, dtype=torch.float32)
    # stacked_patches_rotated = torch.empty(n, N, N, dtype=torch.float32)

    # Fill the 3D tensor with the image data
    # for i, ima'ge_tensor in enumerate(patches_list):
    #     stacked_patches[i] = image_tensor
    # for i, image_tensor in enumerate(rotated_patches):
    #     stacked_patches_rotated[i] = image_tensor
    #
    # tensor_patches = stacked_patches.flatten(1)
    # tensor_patch'es_rotated = stacked_patches_rotated.flatten(1)

    # # Load model
    # N = 18
    # model = AngleNet(N)
    # model_name = 'best_model_1.pt'
    # path = f"saved\{model_name}"
    # model.load_state_dict(torch.load(path), strict=False)
    #
    # angles_test_prediction = model(tensor_patches)  # feedforward of the test images
    # angles_test_prediction_numpy = angles_test_prediction.detach().numpy()  # convert to numpy array (remove gradient)
    #
    # angles_test_prediction_rotated = model(tensor_patches_rotated)
    # angles_test_prediction_numpy_rotated = angles_test_prediction_rotated.detach().numpy()

    # Generate plot
    # fig1, axes1 = create_multiplots(stacked_patches, angles_lines, angles_test_prediction_numpy, number_sample=25)
    # fig2, axes2 = create_multiplots(stacked_patches_rotated, angles_lines_rotated, angles_test_prediction_numpy_rotated, number_sample=25)
    #
    # plt.tight_layout()
    # plt.show()

    # Calculate standard deviation
    # std_dev = calculate_std_dev(angles_lines_normalized, angles_test_prediction_numpy)

    # print('Standard deviation: ', std_dev)
