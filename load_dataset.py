from typing import List
import torch

import matplotlib.pyplot as plt
from numpy import pi
import numpy as np
from random import sample

from classes.diagram_offline import DiagramOffline
from classes.diagram_ndot_offline import DiagramOfflineNDot

from utils.angle_operations import angles_from_list, normalize_angle, get_angle_stat
from utils.settings import settings
from utils.output import init_out_directory, ExistingRunName
from utils.logger import logger
from utils.statistics import resample_dataset

from utils.misc import save_list_to_file
from pathlib import Path
from classes.qdsd import DATA_DIR
from plot.data import plot_patch_sample

run_name = settings.run_name

try:
    # Create the output directory to save results and plots
    init_out_directory()
except ExistingRunName:
    logger.critical(f'Existing run directory: "{run_name}"', exc_info=True)


def load_diagram() -> List["DiagramOffline"]:
    # Load diagrams from files (line and area)
    diagrams = DiagramOfflineNDot.load_diagrams(pixel_size=settings.pixel_size,
                                            research_group=settings.research_group,
                                            diagrams_path=Path(DATA_DIR, 'interpolated_csv.zip'),
                                            labels_path=Path(DATA_DIR, 'labels.json'),
                                            single_dot=True if settings.dot_number == 1 else False,
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

    selected_patches = []
    selected_lines = []

    for patch, line_list in zip(patches_list, lines_list):
        if len(line_list) == 1:
            Dx = np.gradient(patch)[0]
            selected_patches.append(torch.from_numpy(Dx))
            selected_lines.append(line_list)

    # print(lines_list)
    # plot_patch_sample(patches_list, lines_list, sample_number=16, show_offset=True)
    # plot_patch_sample(patches_list, lines_list, sample_number=16, show_offset=True)
    # plot_patch_sample(patches_list, lines_list, sample_number=16, show_offset=False)

    plot_patch_sample(selected_patches, selected_lines, sample_number=16, show_offset=False, name='one_line_DQD_Dx_2')

    # Calculate angles by hand for verification
    angles_lines = angles_from_list(selected_lines)
    angles_lines_normalized = normalize_angle(angles_lines)

    # resampled_patch, resampled_angles, resampled_lines = resample_dataset(selected_patches, angles_lines_normalized, selected_lines, 20)
    # get_angle_stat(resampled_angles)

    # Reshape patches for neural network
    # Get the number of images and the size of each image
    n = len(selected_patches)
    N = selected_patches[0].shape[0]

    # Create an empty tensor with the desired shape
    stacked_patches = torch.empty(n, N, N, dtype=torch.float32)

    # Fill the 3D tensor with the image data
    for i, image_tensor in enumerate(selected_patches):
        stacked_patches[i] = image_tensor

    tensor_patches = stacked_patches.flatten(1)
    print(tensor_patches.shape)
    print(len(angles_lines))
    # Save patches and angles to file for later use
    torch.save(tensor_patches, './saved/double_dot_patches_Dx.pt')
    # save_list_to_file(angles_lines, './saved/double_dot_normalized_angles.txt')
