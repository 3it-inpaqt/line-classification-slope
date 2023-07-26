import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import torch

from classes.diagram_offline import DiagramOffline
from classes.diagram_ndot_offline import DiagramOfflineNDot
from classes.qdsd import DATA_DIR
from plot.data import plot_patch_sample
from plot.lines_visualisation import create_multiplots
from utils.angle_operations import angles_from_list, normalize_angle, get_angle_stat
from utils.logger import logger
from utils.misc import save_list_to_file, renorm_array
from utils.settings import settings
from utils.output import init_out_directory, ExistingRunName

run_name = settings.run_name

try:
    # Create the output directory to save results and plots
    init_out_directory()
except ExistingRunName:
    logger.critical(f'Existing run directory: "{run_name}"', exc_info=True)

# Set LaTex for matplotlib
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
    })


def load_diagram() -> List["DiagramOffline"]:
    # Load diagrams from files (line and area)
    diagrams = DiagramOfflineNDot.load_diagrams(pixel_size=settings.pixel_size,
                                            research_group=settings.research_group,
                                            diagrams_path=Path(DATA_DIR, 'interpolated_csv.zip'),
                                            labels_path=Path(DATA_DIR, 'labels.json'),
                                            single_dot=True if settings.dot_number == 1 else False,
                                            load_lines=True,
                                            load_areas=True,
                                            white_list=None)

    DiagramOfflineNDot.normalize_diagrams(diagrams)

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
        # torch.save(diagram.values[678:696, 138:156], 'test_bastien.pt')  # example of patch being saved for later use
        diagram_patches, lines_patches = diagram.get_patches((settings.patch_size_x, settings.patch_size_y), (6, 6),
                                                             (0, 0))
        patches.extend(diagram_patches)
        lines.extend(lines_patches)
    return patches, lines


if __name__ == '__main__':
    diagrams_exp = load_diagram()
    patches_list, lines_list = load_patches(diagrams_exp)

    selected_patches = []
    selected_lines = []

    for patch, line_list in zip(patches_list, lines_list):
        if len(line_list) == 1:  # takes patch into account only if it has one line crossing it
            if settings.dx:
                Dx = np.gradient(patch)[0]  # derivative with respect to the x-axis
                selected_patches.append(Dx)  # convert numpy array back to torch tensor

            else:
                selected_patches.append(patch)
            selected_lines.append(line_list)

    angles_lines = angles_from_list(selected_lines, normalize=True)
    # get_angle_stat(angles_lines)  # un-comment this line if you want to see the angle statistical distribution

    if settings.rotate_patch:
        from utils.rotation import rotate_patches
        selected_patches, rotated_lines_list, rotated_angle_list = rotate_patches(selected_patches,
                                                                                  selected_lines,
                                                                                  angles_lines)

        get_angle_stat(rotated_angle_list)

    if settings.include_synthetic:
        from utils.populate import populate_angles
        populated_patches, populated_lines_list, populated_angle_list = populate_angles(selected_patches,
                                                                                   selected_lines,
                                                                                   angles_lines,
                                                                                   percentage=0.9,
                                                                                   size=(settings.patch_size_x, settings.patch_size_y),
                                                                                   background=settings.background,
                                                                                   sigma=settings.sigma,
                                                                                   aa=settings.anti_alias)

        get_angle_stat(populated_angle_list)  # plot the angle statistical distribution for the new dataset

        # Plot a sample of patches to see an example of lines
        plot_patch_sample(populated_patches,
                          populated_lines_list,
                          sample_number=16,
                          show_offset=False,
                          name='one_line_populated_DQD')

    # Plot a sample of patches with line highlighted
    plot_patch_sample(selected_patches, selected_lines, sample_number=16, show_offset=False, name='one_line_DQD')

    # Reshape patches for neural network
        # Get the number of images
    n = len(selected_patches)

    # Create an empty tensor with the desired shape
    stacked_patches = torch.empty(n, settings.patch_size_x, settings.patch_size_y, dtype=torch.float32)

    # Fill the tensor with stacked patches
    if type(populated_patches[0]) == np.ndarray:
        stacked_array = np.stack(populated_patches)
        stacked_patches = torch.from_numpy(stacked_array)
    elif type(populated_patches == list):
        for i in range(len(selected_patches)):
            if type(populated_patches[i]) == np.ndarray:
                selected_patch = (populated_patches[i]).copy()  # make a copy of the numpy array
                selected_patch = torch.from_numpy(selected_patch)
                stacked_patches[i, :, :] = selected_patch
            else:
                stacked_patches[i, :, :] = populated_patches[i]
    else:
        stacked_patches = torch.stack(populated_patches)

    tensor_patches = stacked_patches.unsqueeze(1)

    # Set patches and angles path with extra parameters if defined
    path_torch = f'./saved/double_dot_{settings.research_group}_populated_patches_normalized_{settings.patch_size_x}_{settings.patch_size_y}'
    path_angle = f'./saved/double_dot_{settings.research_group}_populated_angles_{settings.patch_size_x}_{settings.patch_size_y}'

    if settings.full_circle:
        path_torch += "_fullcircle"
        path_angle += "_fullcircle"
    if settings.dx:
        path_torch += "_Dx"
        path_angle += "_Dx"

    # Add extension to file path
    path_torch += ".pt"
    path_angle += ".txt"

    # Save tensor
    torch.save(renorm_array(stacked_patches), path_torch)

    # Create multiplot to check some lines
    # fig, axes = create_multiplots(stacked_patches, angles_lines, number_sample=16)  # un-comment this line to see an example of patches
    plt.tight_layout()
    plt.show()

    # Save angles list to file
    save_list_to_file(populated_angle_list, path_angle)  # comment this line out when the patches are all loaded in a tensor, and you only need to apply Dx over them
