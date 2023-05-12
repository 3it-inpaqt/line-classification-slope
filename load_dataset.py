from typing import List

from classes.diagram_offline import DiagramOffline
from utils.settings import settings
from utils.output import init_out_directory, ExistingRunName
from utils.logger import logger
from pathlib import Path
from classes.qdsd import DATA_DIR
from plot.data import plot_samples, plot_patch_sample

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
    patches_multi_line = []
    multi_line_list = []

    for diagram in diagrams:
        diagram_patches, lines_patches = diagram.get_patches((settings.patch_size_x, settings.patch_size_y), (0, 0), (0, 0))
        for patch, lines in zip(diagram_patches, lines_patches):
            if len(lines) == 1:
                patches_one_line.extend(patch)
                one_line_list.extend(lines)
            else:
                for line in lines:
                    patches_multi_line.extend(patch)
                    multi_line_list.extend(line)

    return patches_one_line, one_line_list, patches_multi_line, multi_line_list


if __name__ == '__main__':
    diagrams_exp = load_diagram()
    patches_list = load_patches(diagrams_exp)

    # samples_patches = [patch for patch in patches_list[::5]]

