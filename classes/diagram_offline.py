import gzip
import json
import zipfile
from pathlib import Path
import platform
from random import randrange
from typing import Any, IO, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from shapely.geometry import LineString, Point, Polygon

from classes.data_structures import ChargeRegime
from classes.diagram import Diagram
from plot.data import plot_diagram
from utils.logger import logger
from utils.output import load_normalization, save_normalization
from utils.settings import settings


class DiagramOffline(Diagram):
    """ Handle the diagram data and its annotations. """

    # The list of voltage for the first gate
    x_axes: Sequence[float]

    # The list of voltage for the second gate
    y_axes: Sequence[float]

    # The list of measured voltage according to the 2 gates
    values: torch.Tensor

    # The transition lines annotations
    transition_lines: Optional[List[LineString]]

    # The charge area lines annotations
    charge_areas: Optional[List[Tuple[type(ChargeRegime), Polygon]]]

    def __init__(self, file_basename: str, x_axes: Sequence[float], y_axes: Sequence[float], values: torch.Tensor,
                 transition_lines: Optional[List[LineString]],
                 charge_areas: Optional[List[Tuple[type(ChargeRegime), Polygon]]]):
        super().__init__(file_basename)

        self.x_axes = x_axes
        self.y_axes = y_axes
        self.values = values
        self.transition_lines = transition_lines
        self.charge_areas = charge_areas

    def get_random_starting_point(self) -> Tuple[int, int]:
        """
        Generate (pseudo) random coordinates for the top left corner of a patch inside the diagram.
        :return: The (pseudo) random coordinates.
        """
        max_x, max_y = self.get_max_patch_coordinates()
        return randrange(max_x), randrange(max_y)

    def voltage_to_coord_x(self, x: float) -> Tuple[int, int]:
        """
        Convert a voltage to a coordinate in the diagram relatively to the origin chosen.

        :param x: The voltage x-axis to convert.
        :return: The coordinate x in the diagram.
        """
        return round(((x - self.x_axes[0]) / settings.pixel_size))

    def voltage_to_coord_y(self, y: float) -> Tuple[int, int]:
        """
        Convert a voltage to a coordinate in the diagram relatively to the origin chosen.

        :param y: The voltage y-axis to convert.
        :return: The coordinate y in the diagram.
        """
        return round(((y - self.y_axes[0]) / settings.pixel_size))

    def coord_to_voltage(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert a coordinate in the diagram to a voltage.

        :param x: The coordinate (x axes) to convert.
        :param y: The coordinate (y axes) to convert.
        :return: The voltage (x, y) in this diagram.
        """
        x_volt = self.x_axes[0] + x * settings.pixel_size
        y_volt = self.y_axes[0] + y * settings.pixel_size
        return x_volt, y_volt

    def get_patch(self, coordinate: Tuple[int, int], patch_size: Tuple[int, int]) -> torch.Tensor:
        """
        Extract one patch in the diagram (data only, no label).

        :param coordinate: The coordinate in the diagram (not the voltage)
        :param patch_size: The size of the patch to extract (in number of pixel)
        :return: The patch
        """
        coord_x, coord_y = coordinate
        size_x, size_y = patch_size

        diagram_size_y, _ = self.values.shape
        end_y = coord_y + size_y
        end_x = coord_x + size_x

        # Invert Y axis because the diagram origin (0,0) is top left
        return self.values[diagram_size_y - end_y:diagram_size_y - coord_y, coord_x:end_x]

    def get_patches(self, patch_size: Tuple[int, int] = (10, 10), overlap: Tuple[int, int] = (0, 0),
                    label_offset: Tuple[int, int] = (0, 0)) -> Any:
        """
        Create patches from diagrams sub-area.

        :param patch_size: The size of the desired patches, in number of pixels (x, y)
        :param overlap: The size of the patches overlapping, in number of pixels (x, y)
        :param label_offset: The width of the border to ignore during the patch labeling, in number of pixel (x, y)
        :return: A tuple of patches and associated intersecting lines.
        """
        patch_size_x, patch_size_y = patch_size
        overlap_size_x, overlap_size_y = overlap
        label_offset_x, label_offset_y = label_offset
        # print(self.values.shape)
        diagram_size_y, diagram_size_x = self.values.shape

        patches_intersected = []  # contains the patch where there is at least one line intersecting
        lines_intersecting = []  # contains the lines that intersect patch with corresponding index in list patches_intersected

        # Extract each patches
        i = 0
        for patch_y in range(0, diagram_size_y - patch_size_y, patch_size_y - overlap_size_y):
            # Patch coordinates (indexes)
            start_y = patch_y
            end_y = patch_y + patch_size_y
            # Patch coordinates (voltage)
            start_y_v = self.y_axes[start_y + label_offset_y]
            end_y_v = self.y_axes[end_y - label_offset_y]
            for patch_x in range(0, diagram_size_x - patch_size_x, patch_size_x - overlap_size_x):
                i += 1
                # Patch coordinates (indexes)
                start_x = patch_x
                end_x = patch_x + patch_size_x
                # Patch coordinates (voltage) for label area
                start_x_v = self.x_axes[start_x + label_offset_x]
                end_x_v = self.x_axes[end_x - label_offset_x]

                # Create patch shape to find line intersection
                patch_shape = Polygon([(start_x_v, start_y_v),
                                       (end_x_v, start_y_v),
                                       (end_x_v, end_y_v),
                                       (start_x_v, end_y_v)])

                # Extract patch value
                # Invert Y axis because the diagram origin (0,0) is top left
                patch = self.values[diagram_size_y - end_y:diagram_size_y - start_y, start_x:end_x]
                # print(patch.shape)

                # Find all the lines intersecting the patch
                for lines in self.transition_lines:
                    patch_intersecting_lines = [line for line in lines if line.intersects(patch_shape)]
                    for line in patch_intersecting_lines:
                        x_line, y_line = line.xy
                        segments = [LineString(zip(x_line[i:i+2], y_line[i:i+2])) for i in range(0, len(x_line) - 1)]
                        # Divide the line in segments and check for each of they intersect the patch to reduce the
                        # size of the intersecting lines and only take effective crossing
                        segments_intersecting = [([self.voltage_to_coord_x(x) - patch_x for x in segment.xy[0]],
                                                  [self.voltage_to_coord_y(y) - patch_y for y in segment.xy[1]])
                                                 for segment in segments if segment.intersects(patch_shape)]
                        # print('patch x: ', patch_x)
                        # print('patch y: ', patch_y)
                        # print(segments_intersecting)
                        # print('test x coordinates: ', segments[0].xy[0])
                        # print('test x conversion: ', segments_intersecting[0][0])
                        # print('-----------------------------------')
                        lines_intersecting.append(segments_intersecting)
                        patches_intersected.append(patch)

        return patches_intersected, lines_intersecting

    def get_charge(self, coord_x: int, coord_y: int) -> ChargeRegime:
        """
        Get the charge regime of a specific location in the diagram.

        :param coord_x: The x coordinate to check (not the voltage)
        :param coord_y: The y coordinate to check (not the voltage)
        :return: The charge regime
        """
        volt_x = self.x_axes[coord_x]
        volt_y = self.y_axes[coord_y]
        point = Point(volt_x, volt_y)

        # Check coordinates in each labeled area
        for regime, area in self.charge_areas:
            if area.contains(point):
                return regime

        # Coordinates not found in labeled areas. The charge area in this location is thus unknown.
        return ChargeRegime.UNKNOWN

    def is_line_in_patch(self, coordinate: Tuple[int, int],
                         patch_size: Tuple[int, int],
                         offsets: Tuple[int, int] = (0, 0)) -> bool:
        """
        Check if a line label intersect a specific sub-area (patch) of the diagram.

        :param coordinate: The patch top left coordinates
        :param patch_size: The patch size
        :param offsets: The patch offset (area to ignore lines)
        :return: True if a line intersect the patch (offset excluded)
        """

        coord_x, coord_y = coordinate
        size_x, size_y = patch_size
        offset_x, offset_y = offsets

        # Subtract the offset and convert to voltage
        start_x_v = self.x_axes[coord_x + offset_x]
        start_y_v = self.y_axes[coord_y + offset_y]
        end_x_v = self.x_axes[coord_x + size_x - offset_x]
        end_y_v = self.y_axes[coord_y + size_y - offset_y]

        # Create patch shape to find line intersection
        patch_shape = Polygon([(start_x_v, start_y_v),
                               (end_x_v, start_y_v),
                               (end_x_v, end_y_v),
                               (start_x_v, end_y_v)])

        # Label is True if any line intersect the patch shape
        return any([line.intersects(patch_shape) for line in self.transition_lines])

    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False,
           copy: bool = False):
        """
        Send the dataset to a specific device (cpu or cuda) and/or a convert it to a different type.
        Modification in place.
        The arguments correspond to the torch tensor "to" signature.
        See https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to.
        """
        self.values = self.values.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)

    def get_values(self) -> Tuple[Optional[torch.Tensor], Sequence[float], Sequence[float]]:
        """
        Get the values of the diagram and the corresponding axis.

        :return: The values as a tensor, the list of x-axis values, the list of y-axis values
        """
        return self.values.detach().cpu(), self.x_axes, self.y_axes

    def plot(self, focus_area: Optional[Tuple] = None, label_extra: Optional[str] = '') -> None:
        """
        Plot the diagram with matplotlib (save and/or show it depending on the settings).
        This method is a shortcut of plots.diagram.plot_diagram.

        :param focus_area: Optional coordinates to restrict the plotting area. A Tuple as (x_min, x_max, y_min, y_max).
        :param label_extra: Optional extra information for the plot label.
        """
        plot_diagram(self.x_axes, self.y_axes, self.values, self.file_basename + label_extra, 'nearest',
                     self.x_axes[1] - self.x_axes[0], transition_lines=self.transition_lines,
                     charge_regions=self.charge_areas, focus_area=focus_area, show_offset=False, scale_bar=True)

    def get_max_patch_coordinates(self) -> Tuple[int, int]:
        """
        Get the maximum coordinates of a patch in this diagram.

        :return: The maximum coordinates as (x, y)
        """
        return len(self.x_axes) - settings.patch_size_x - 1, len(self.y_axes) - settings.patch_size_y - 1

    def __str__(self):
        return '[OFFLINE] ' + super().__str__() + f' (size: {len(self.x_axes)}x{len(self.y_axes)})'

    @staticmethod
    def load_diagrams(pixel_size,
                      research_group,
                      diagrams_path: Path,
                      labels_path: Path = None,
                      single_dot: bool = True,
                      load_lines: bool = True,  #
                      load_areas: bool = True,
                      white_list: List[str] = None) -> List["DiagramOffline"]:
        """
        Load stability diagrams and annotations from files.

        :param pixel_size: The size of one pixel in volt
        :param research_group: The research_group name for the dataset to load
        :param single_dot: If True, only the single dot diagram will be loaded, if False only the double dot
        :param diagrams_path: The path to the zip file containing all stability diagrams data.
        :param labels_path: The path to the json file containing line and charge area labels.
        :param load_lines: If True the line labels should be loaded.
        :param load_areas: If True the charge area labels should be loaded.
        :param white_list: If defined, only diagrams with base name include in this list will be loaded (no extension).
        :return: A list of Diagram objects.
        """

        # Open the json file that contains annotations for every diagrams
        with open(labels_path, 'r') as annotations_file:
            labels_json = json.load(annotations_file)

        logger.debug(f'{len(labels_json)} labeled diagrams found')
        labels = {obj['External ID']: obj for obj in labels_json}

        # Open the zip file and iterate over all csv files
        # in_zip_path should use "/" separator, no matter the current OS
        in_zip_path = f'{pixel_size * 1000}mV/' + ('single' if single_dot else 'double') + f'/{research_group}/'
        zip_dir = zipfile.Path(diagrams_path, at=in_zip_path)

        if not zip_dir.is_dir():
            raise ValueError(f'Folder "{in_zip_path}" not found in the zip file "{diagrams_path}".'
                             f'Check if pixel size and research group exist in this folder.')

        diagrams = []
        nb_no_label = 0
        nb_excluded = 0
        # Iterate over all csv files inside the zip file
        for diagram_name in zip_dir.iterdir():
            file_basename = Path(str(diagram_name)).stem  # Remove extension

            if white_list and not (file_basename in white_list):
                nb_excluded += 1
                continue

            if f'{file_basename}.png' not in labels:
                logger.debug(f'No label found for {file_basename}')
                nb_no_label += 1
                continue

            # Windows needs the 'b' option
            open_options = 'rb' if platform.system() == 'Windows' else 'r'
            with diagram_name.open(open_options) as diagram_file:
                # Load values from CSV file
                x, y, values = DiagramOffline._load_interpolated_csv(gzip.open(diagram_file))

                current_labels = labels[f'{file_basename}.png']['Label']
                label_pixel_size = float(next(filter(lambda l: l['title'] == 'pixel_size_volt',
                                                     current_labels['classifications']))['answer'])
                transition_lines = None
                charge_area = None

                if load_lines:
                    # Load transition line annotations
                    transition_lines = DiagramOffline._load_lines_annotations(
                        filter(lambda l: l['title'] == 'line_1', current_labels['objects']), x, y,
                        pixel_size=label_pixel_size,
                        snap=1)

                    if len(transition_lines) == 0:
                        logger.debug(f'No line label found for {file_basename}')
                        nb_no_label += 1
                        continue

                if load_areas:
                    # TODO adapt for double dot (load N_electron_2 too)
                    # Load charge area annotations
                    charge_area = DiagramOffline._load_charge_annotations(
                        filter(lambda l: l['title'] != 'line_1', current_labels['objects']), x, y,
                        pixel_size=label_pixel_size,
                        snap=1)

                    if len(charge_area) == 0:
                        logger.debug(f'No charge label found for {file_basename}')
                        nb_no_label += 1
                        continue

                diagram = DiagramOffline(file_basename, x, y, values, transition_lines, charge_area)
                diagrams.append(diagram)
                if settings.plot_diagrams:
                    diagram.plot()

        if nb_no_label > 0:
            logger.warning(f'{nb_no_label} diagram(s) skipped because no label found')

        if nb_excluded > 0:
            logger.info(f'{nb_excluded} diagram(s) excluded because not in white list')

        if len(diagrams) == 0:
            logger.error(f'No diagram loaded in "{zip_dir}"')

        return diagrams

    @staticmethod
    def _load_interpolated_csv(file_path: Union[IO, str, Path]) -> Tuple:
        """
        Load the stability diagrams from CSV file.

        :param file_path: The path to the CSV file or the byte stream.
        :return: The stability diagram data as a tuple: x, y, values
        """
        compact_diagram = np.loadtxt(file_path, delimiter=',')
        # Extract information
        x_start, y_start, step = compact_diagram[0][0], compact_diagram[0][1], compact_diagram[0][2]

        # Remove the information row
        values = np.delete(compact_diagram, 0, 0)

        # Reconstruct the axes
        x = np.arange(values.shape[1]) * step + x_start
        y = np.arange(values.shape[0]) * step + y_start

        return x, y, torch.tensor(values, dtype=torch.float)

    @staticmethod
    def _load_lines_annotations(lines: Iterable, x, y, pixel_size: float, snap: int = 1) -> List[LineString]:
        """
        Load transition line annotations for an image.

        :param lines: List of line label as json object (from Labelbox export)
        :param x: The x-axis of the diagram (in volt)
        :param y: The y-axis of the diagram (in volt)
        :param pixel_size: The pixel size for these labels (as a ref ton convert axes to volt)
        :param snap: The snap margin, every points near to image border at this distance will be rounded to the image
         border (in number of pixels)
        :return: The list of line annotation for the image, as shapely.geometry.LineString
        """
        # TODO Change to have separated lines coordinates (x1, x2), (y1, y2) and not a long tuple
        processed_lines = []
        for line in lines:
            line_x = DiagramOffline._coord_to_volt((p['x'] for p in line['line']), x[0], x[-1], pixel_size, snap)
            line_y = DiagramOffline._coord_to_volt((p['y'] for p in line['line']), y[0], y[-1], pixel_size, snap, True)
            line_obj = LineString(zip(line_x, line_y))
            processed_lines.append(line_obj)

        return processed_lines

    @staticmethod
    def _load_charge_annotations(charge_areas: Iterable, x, y, pixel_size: float, snap: int = 1) \
            -> List[Tuple[type(ChargeRegime), Polygon]]:
        """
        Load regions annotation for an image.

        :param charge_areas: List of charge area label as json object (from Labelbox export)
        :param x: The x-axis of the diagram (in volt)
        :param y: The y-axis of the diagram (in volt)
        :param pixel_size: The pixel size for these labels (as a ref ton convert axes to volt)
        :param snap: The snap margin, every points near to image border at this distance will be rounded to the image
        border (in number of pixels)
        :return: The list of regions annotation for the image, as (label, shapely.geometry.Polygon)
        """

        processed_areas = []
        for area in charge_areas:
            area_x = DiagramOffline._coord_to_volt((p['x'] for p in area['polygon']), x[0], x[-1], pixel_size, snap)
            area_y = DiagramOffline._coord_to_volt((p['y'] for p in area['polygon']), y[0], y[-1], pixel_size, snap,
                                                   True)

            area_obj = Polygon(zip(area_x, area_y))
            processed_areas.append((ChargeRegime[area['title']], area_obj))
        return processed_areas

    @staticmethod
    def normalize_diagrams(diagrams: Iterable["DiagramOffline"]) -> None:
        """
        Re-normalise a tensor with value between 0 and 1
        :param diagrams: Diagrams object
        :return: Normalized tensor of size [1, N, N]
        """
        # Concatenate all the tensors into a single tensor
        # print([diagram.values.shape for diagram in diagrams])
        all_max = [diagram.values.max().item() for diagram in diagrams]
        all_min = [diagram.values.min().item() for diagram in diagrams]

        # Compute the minimum and maximum values across all the tensors
        min_value = min(all_min)
        max_value = max(all_max)
        # print(max_value)
        # print(min_value)

        save_normalization(min_value=min_value, max_value=max_value)

        # Normalize each tensor in the list using the computed minimum and maximum values
        for diagram in diagrams:
            tensor = diagram.values.clone()
            new_tensor = tensor.view(1, -1)
            new_tensor -= min_value
            new_tensor /= max_value
            diagram.values = new_tensor.view(tensor.size(0), tensor.size(1))



