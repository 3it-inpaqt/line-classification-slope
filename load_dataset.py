from typing import List

from classes.diagram_offline import DiagramOffline
from utils.settings import settings
from utils.output import init_out_directory, ExistingRunName
from utils.logger import logger
from pathlib import Path
from classes.qdsd import DATA_DIR
from plot.data import plot_patch_sample, plot_samples

import matplotlib.pyplot as plt

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
    patches_one_line = []
    one_line_list = []
    # Patches with several lines
    # patches_multi_line = []
    # multi_line_list = []

    for diagram in diagrams:
        diagram_patches, lines_patches = diagram.get_patches((settings.patch_size_x, settings.patch_size_y), (6, 6),
                                                             (0, 0))
        # print('len ', len(diagram_patches))
        # print(lines_patches)
        for patch, lines in zip(diagram_patches, lines_patches):
            # print(patch.shape)
            # print(len(lines))
            # if len(lines) == 1:
            patches_one_line.append(patch)
            one_line_list.append(lines)
            # print(len(patches_one_line))
            # print(len(one_line_list))
            # else:
            #     for line in lines:
            #         patches_multi_line.append(patch)
            #         multi_line_list.append(line)
            # break

        # break
        # print(patch)
    # print(one_line_list[318])
    return patches_one_line, one_line_list  # , patches_multi_line, multi_line_list


if __name__ == '__main__':
    diagrams_exp = load_diagram()
    print('diagram ', len(diagrams_exp))
    patches_one_line, one_line_list = load_patches(diagrams_exp)
    # print(len(patches_one_line))
    # print(one_line_list)
    plot_patch_sample(patches_one_line, one_line_list, sample_number=25)
    # plot_samples(patches_one_line, one_line_list, title='Patches', file_name='test')

    """ Test plot for one patch """
    # index = 0  # visible line by eye
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # line = one_line_list[index]
    # patch = patches_one_line[index]
    # # print(line)
    # # height, width = patch.shape[:2]  # Get the height and width of the image
    # # ax.set_xlim([0, width])  # Set the x-axis limits to match the width of the image
    # # ax.set_ylim([height, 0])  # Set the y-axis limits to match the height of the image (note the inverted y-axis)
    #
    # x_lim, y_lim = line[0], line[1]
    # # print(x_lim, y_lim)
    #
    # ax.plot(x_lim, y_lim, color='blue')
    # ax.imshow(patch, interpolation='nearest', cmap='copper')
    #
    # ax.set_xlim(0, settings.patch_size_x)
    # ax.set_ylim(0, settings.patch_size_y)
    # ax.axis('off')
    #
    # plt.show()
