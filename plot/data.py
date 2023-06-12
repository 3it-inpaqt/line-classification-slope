import io
from copy import copy
# from functools import partial
from math import ceil, sqrt
from random import sample
# from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.legend_handler import HandlerBase
from shapely.geometry import LineString, Polygon
import torch
# from torch.utils.data import DataLoader, Dataset

# from utils.misc import get_nb_loader_workers
from utils.output import save_plot
from utils.settings import settings
from utils.logger import logger
from utils.angle_operations import calculate_angle, normalize_angle

LINE_COLOR = 'blue'
NO_LINE_COLOR = 'tab:red'
GOOD_COLOR = 'green'
ERROR_COLOR = 'tab:red'
SOFT_ERROR_COLOR = 'blueviolet'
UNKNOWN_COLOR = 'dimgray'
NOT_SCANNED_COLOR = 'lightgrey'


def plot_diagram(x_i, y_i,
                 pixels: Optional,
                 image_name: str,
                 interpolation_method: Optional[str],
                 pixel_size: float,
                 charge_regions: Iterable[Tuple["ChargeRegime", Polygon]] = None,
                 transition_lines: Iterable[LineString] = None,
                 focus_area: Optional[Tuple] = None,
                 show_offset: bool = True,
                 scan_history: List["StepHistoryEntry"] = None,
                 scan_errors: bool = False,
                 confidence_thresholds: List[float] = None,
                 fog_of_war: bool = False,
                 fading_history: int = 0,
                 history_uncertainty: bool = False,
                 scale_bar: bool = False,
                 final_coord: Tuple[int, int] = None,
                 save_in_buffer: bool = False,
                 text_stats: bool = False,
                 show_title: Optional[bool] = None,
                 show_crosses: bool = True,
                 vmin: float = None,
                 vmax: float = None,
                 allow_overwrite: bool = False) -> Optional[Union[Path, io.BytesIO]]:
    """
    Plot the interpolated image. This function is a multi-tool nightmare.

    :param x_i: The x coordinates of the pixels (post interpolation).
    :param y_i: The y coordinates of the pixels (post interpolation).
    :param pixels: The list of pixels to plot.
    :param image_name: The name of the image, used for plot title and file name.
    :param interpolation_method: The pixels' interpolation method used to process the pixels,
        used as information in the title.
    :param pixel_size: The size of pixels, in voltage, used for plot title.
    :param charge_regions: The charge region annotations to draw on top of the image.
    :param transition_lines: The transition line annotation to draw on top of the image.
    :param focus_area: Optional coordinates to restrict the plotting area. A Tuple as (x_min, x_max, y_min, y_max).
    :param show_offset: If True, draw the offset rectangle (ignored if both offset x and y are 0).
    :param scan_history: The tuning steps history (see StepHistoryEntry dataclass).
    :param scan_errors: If True, and scan_history defined, plot the step error on the diagram. If False plot the class
        inference instead. Soft errors are shown only if uncertainty is disabled.
    :param confidence_thresholds: The model confidence threshold values for each class. Only necessary if scan_errors
     and not history_uncertainty enabled (yes, this is very specific).
    :param fog_of_war: If True, and scan_history defined, hide the section of the diagram that was never scanned.
    :param fading_history: The number of scan inference the plot, the latest first. The number set will be plotted with
        solid color and the same number will fad progressively. Not compatible with history_uncertainty.
    :param history_uncertainty: If True and scan_history provided, plot steps with full squares and alpha representing
        the uncertainty.
    :param scale_bar: If True, and pixels provided, plot the pixel color scale at the right of the diagram. If the data
        are normalized this scale unit doesn't make sense.
    :param final_coord: The final tuning coordinates.
    :param save_in_buffer: If True, save the image in memory. Do not plot or save it on the disk.
    :param text_stats: If True, add statistics information in the plot.
    :param show_title: If True, plot figure title. If omitted, show title only if not latex format.
    :param show_crosses: If True, plot the crosses representing the start and the end of the tuning if possible.
    :param vmin: Minimal pixel value for color scaling. Set to keep consistant color between plots. If None, the scaling
        is computed by matplotlib based on pixel currently visible.
    :param vmax: Maximal pixel value for color scaling. Set to keep consistant color between plots. If None, the
        scaling is computed by matplotlib based on pixel currently visible.
    :param allow_overwrite: If True, allow to overwrite existing plot.
    :return: The path where the plot is saved, or None if not saved. If save_in_buffer is True, return image bytes
        instead of the path.
    """

    legend = False
    # By default do not plot title for latex format.
    show_title = not settings.image_latex_format if show_title is None else show_title

    with sns.axes_style("ticks"):  # Temporary change the axe style (avoid white ticks)
        boundaries = [np.min(x_i), np.max(x_i), np.min(y_i), np.max(y_i)]
        if pixels is None:
            # If no pixels provided, plot a blank image to allow other information on the same format
            plt.imshow(np.zeros((len(x_i), len(y_i))), cmap=LinearSegmentedColormap.from_list('', ['white', 'white']),
                       extent=boundaries)
        else:
            if fog_of_war and scan_history is not None and len(scan_history) > 0:
                # Mask area not scanned
                mask = np.full_like(pixels, True)
                for scan in scan_history:
                    x, y = scan.coordinates
                    y = len(y_i) - y  # Origin to bottom left
                    mask[y - settings.patch_size_y: y, x:x + settings.patch_size_x] = False
                pixels = np.ma.masked_array(pixels, mask)

            cmap = matplotlib.cm.copper
            cmap.set_bad(color=NOT_SCANNED_COLOR)
            plt.imshow(pixels, interpolation='nearest', cmap=cmap, extent=boundaries, vmin=vmin, vmax=vmax)
            if scale_bar:
                if settings.research_group == 'michel_pioro_ladriere' or \
                        settings.research_group == 'eva_dupont_ferrier':
                    measuring = r'$\mathregular{I_{SET}}$'
                elif settings.research_group == 'louis_gaudreau':
                    measuring = r'$\mathregular{I_{QPC}}$'
                else:
                    measuring = 'I'
                plt.colorbar(shrink=0.85, label=f'{measuring} (A)')

    charge_text = None  # Keep on text field for legend
    if charge_regions is not None:
        for regime, polygon in charge_regions:
            polygon_x, polygon_y = polygon.exterior.coords.xy
            plt.fill(polygon_x, polygon_y, facecolor=(0, 0, 0.5, 0.3), edgecolor=(0, 0, 0.5, 0.8), snap=True)
            label_x, label_y = list(polygon.centroid.coords)[0]
            charge_text = plt.text(label_x, label_y, str(regime), ha="center", va="center", color='b', weight='bold',
                                   bbox=dict(boxstyle='round', pad=0.2, facecolor='w', alpha=0.5, edgecolor='w'))
    # print(transition_lines)
    if transition_lines is not None:
        for i, line in enumerate(transition_lines):
            for l in line:
                line_x, line_y = l.xy[0], l.xy[1]
                plt.plot(line_x, line_y, color='lime')
            legend = True

    if scan_history is not None and len(scan_history) > 0:
        from classes.qdsd import QDSDLines  # Import here to avoid circular import
        first_patch_label = set()

        patch_size_x_v = (settings.patch_size_x - settings.label_offset_x * 2) * pixel_size
        patch_size_y_v = (settings.patch_size_y - settings.label_offset_y * 2) * pixel_size

        for i, scan_entry in enumerate(reversed(scan_history)):
            line_detected = scan_entry.model_classification
            x, y = scan_entry.coordinates
            alpha = 1

            if scan_errors:
                # Patch color depending on the classification success
                if not history_uncertainty and scan_entry.is_under_confidence_threshold(confidence_thresholds):
                    # If the uncertainty is not shown with alpha, we show it by a gray patch
                    color = UNKNOWN_COLOR
                    label = 'Unknown'
                elif scan_entry.is_classification_correct():
                    color = GOOD_COLOR
                    label = 'Good'
                elif not history_uncertainty and scan_entry.is_classification_almost_correct():
                    # Soft error is not compatible with uncertainty
                    color = SOFT_ERROR_COLOR
                    label = 'Soft Error'
                else:
                    color = ERROR_COLOR
                    label = 'Error'
            else:
                # Patch color depending on the inferred class
                color = LINE_COLOR if line_detected else NO_LINE_COLOR
                label = f'Infer {QDSDLines.classes[line_detected]}'

            # Add label only if it is the first time we plot a patch with this label
            if label in first_patch_label:
                label = None
            else:
                first_patch_label.add(label)

            if history_uncertainty or fading_history == 0 or i < fading_history * 2:  # Condition to plot patches
                if history_uncertainty:
                    # Transparency based on the confidence
                    alpha = scan_entry.model_confidence
                    label = None  # No label since we have the scale bar
                elif fading_history == 0 or i < fading_history * 2:
                    if fading_history != 0:
                        # History fading for fancy animation
                        alpha = 1 if i < fading_history else (2 * fading_history - i) / (fading_history + 1)
                    legend = True

                if pixels is None:
                    # Full patch if white background
                    face_color = color
                    edge_color = 'none'
                else:
                    # Empty patch if diagram background
                    face_color = 'none'
                    edge_color = color

                patch = patches.Rectangle((x_i[x + settings.label_offset_x], y_i[y + settings.label_offset_y]),
                                          patch_size_x_v,
                                          patch_size_y_v,
                                          linewidth=1,
                                          edgecolor=edge_color,
                                          label=label,
                                          facecolor=face_color,
                                          alpha=alpha)
                plt.gca().add_patch(patch)

        # Marker for first point
        if show_crosses and (fading_history == 0 or len(scan_history) < fading_history * 2):
            first_x, first_y = scan_history[0].coordinates
            if fading_history == 0:
                alpha = 1
            else:
                # Fading after the first scans if fading_history is enabled
                i = len(scan_history) - 2
                alpha = 1 if i < fading_history else (2 * fading_history - i) / (fading_history + 1)
            plt.scatter(x=x_i[first_x + settings.patch_size_x // 2], y=y_i[first_y + settings.patch_size_y // 2],
                        color='skyblue', marker='X', s=200, label='Start', alpha=alpha)
            legend = True

        if history_uncertainty:
            # Set up the color-bar
            if scan_errors:
                cmap = LinearSegmentedColormap.from_list('', [GOOD_COLOR, 'white', ERROR_COLOR])
            else:
                cmap = LinearSegmentedColormap.from_list('', [LINE_COLOR, 'white', ERROR_COLOR])
            norm = Normalize(vmin=-1, vmax=1)
            cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), shrink=0.8, aspect=15)
            cbar.outline.set_edgecolor('0.15')
            cbar.set_ticks([-1, 0, 1])

            # Bayesian uncertainty
            if settings.model_type.upper() in ['BCNN', 'BFF']:
                metric_map = {  # This plot is not compatible with not normalized uncertainty
                    'norm_std': 'Normalized STD',
                    'norm_entropy': 'Normalized entropy'
                }
                uncertainty_label = metric_map[settings.bayesian_confidence_metric]
                min_uncertainty_correct = min_uncertainty_line = min_uncertainty_no_line = 0
                max_uncertainty = 1

            # Ad hoc uncertainty
            else:
                uncertainty_label = 'Model output'
                min_uncertainty_line = 1
                min_uncertainty_no_line = 0
                min_uncertainty_correct = f'{min_uncertainty_line} or {min_uncertainty_no_line}'
                max_uncertainty = 0.5

            if scan_errors:
                cbar.set_ticklabels(
                    [f'Correct class\n{uncertainty_label}: {min_uncertainty_correct}\n(Low uncertainty)',
                     f'{uncertainty_label}: {max_uncertainty}\n(High uncertainty)',
                     f'Error class\n{uncertainty_label}: {min_uncertainty_correct}\n(Low uncertainty)'])
            else:
                cbar.set_ticklabels([f'Line\n{uncertainty_label}: {min_uncertainty_line}\n(Low uncertainty)',
                                     f'{uncertainty_label}: {max_uncertainty}\n(High uncertainty)',
                                     f'No Line\n{uncertainty_label}: {min_uncertainty_no_line}\n(Low uncertainty)'])

    # Marker for tuning final guess
    if show_crosses and final_coord is not None:
        last_x, last_y = final_coord
        # Get marker position (and avoid going out)
        last_x_i = min(last_x, len(x_i) - 1)
        last_y_i = min(last_y, len(y_i) - 1)
        plt.scatter(x=x_i[last_x_i], y=y_i[last_y_i], color='w', marker='x', s=210, linewidths=2)  # Make white borders
        plt.scatter(x=x_i[last_x_i], y=y_i[last_y_i], color='fuchsia', marker='x', s=200, label='End')
        legend = True

    if show_offset and (settings.label_offset_x != 0 or settings.label_offset_y != 0):
        focus_x, focus_y = focus_area if focus_area else 0, 0

        # Create a Rectangle patch
        rect = patches.Rectangle((x_i[settings.label_offset_x] - pixel_size * 0.35,
                                  y_i[settings.label_offset_y] - pixel_size * 0.35),
                                 (focus_x + settings.patch_size_x - 2 * settings.label_offset_x) * pixel_size,
                                 (focus_y + settings.patch_size_y - 2 * settings.label_offset_y) * pixel_size,
                                 linewidth=1.5, edgecolor='fuchsia', facecolor='none')

        # Add the patch to the Axes
        plt.gca().add_patch(rect)

    if text_stats:
        text = ''
        if scan_history and len(scan_history) > 0:
            # Local import to avoid circular mess
            from classes.qdsd import QDSDLines

            accuracy = sum(1 for s in scan_history if s.is_classification_correct()) / len(scan_history)
            nb_line = sum(1 for s in scan_history if s.ground_truth)  # s.ground_truth == True means line
            nb_no_line = sum(1 for s in scan_history if not s.ground_truth)  # s.ground_truth == False means no line

            if nb_line > 0:
                line_success = sum(
                    1 for s in scan_history if s.ground_truth and s.is_classification_correct()) / nb_line
            else:
                line_success = None

            if nb_no_line > 0:
                no_line_success = sum(1 for s in scan_history
                                      if not s.ground_truth and s.is_classification_correct()) / nb_no_line
            else:
                no_line_success = None

            if scan_history[-1].is_classification_correct():
                class_error = 'good'
            elif scan_history[-1].is_classification_almost_correct():
                class_error = 'soft error'
            else:
                class_error = 'error'
            last_class = QDSDLines.classes[scan_history[-1].model_classification]

            text += f'Nb step: {len(scan_history): >3n} (acc: {accuracy: >4.0%})\n'
            text += f'{QDSDLines.classes[True].capitalize(): <7}: {nb_line: >3n}'
            text += '\n' if line_success is None else f' (acc: {line_success:>4.0%})\n'
            text += f'{QDSDLines.classes[False].capitalize(): <7}: {nb_no_line: >3n}'
            text += '\n' if no_line_success is None else f' (acc: {no_line_success:>4.0%})\n\n'
            text += f'Last scan:\n'
            text += f'  - Pred: {last_class.capitalize(): <7} ({class_error})\n'
            text += f'  - Conf: {scan_history[-1].model_confidence: >4.0%}\n\n'
            text += f'Tuning step:\n'
            text += f'  {scan_history[-1].description}'

        plt.text(1.03, 0.8, text, horizontalalignment='left', verticalalignment='top', fontsize=8,
                 fontfamily='monospace', transform=plt.gca().transAxes)

    if show_title:
        interpolation_str = f'interpolated ({interpolation_method}) - ' if interpolation_method is not None else ''
        plt.title(f'{image_name}\n{interpolation_str}pixel size {round(pixel_size, 10) * 1_000}mV')

    plt.xlabel('G1 (V)')
    plt.xticks(rotation=30)
    plt.ylabel('G2 (V)')

    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        handler_map = None
        if charge_text is not None:
            # Create custom legend for charge regime text
            charge_text = copy(charge_text)
            charge_text.set(text='N')
            handler_map = {type(charge_text): TextHandler()}
            handles.append(charge_text)
            labels.append('Charge regime')

        plt.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.35), handles=handles, labels=labels,
                   handler_map=handler_map)

    if focus_area:
        plt.axis(focus_area)

    return save_plot(f'diagram_{image_name}', allow_overwrite=allow_overwrite, save_in_buffer=save_in_buffer)


def plot_patch_sample(patches_list: List[torch.Tensor], lines_list: List[Any], sample_number: int, show_offset: bool = True, name: str = 'patch_sample') -> None:
    """
    Plot randomly sampled patches grouped by class.

    :param name: File name
    :param patches_list: The patches list to sample from.
    :param lines_list: List of the lines intersecting the patches
    :param sample_number: The number of patches to sample
    :param show_offset: If True draws the offset rectangle (ignored if both offset x and y are 0)
    """
    # Check if sample number is not greater than the amount of available data
    if sample_number > len(lines_list):
        sample_number = len(lines_list)
        logger.warning(f'{len(lines_list)} diagrams available but number of sampled diagram was set to {sample_number}. Only {len(lines_list)} will be sampled')

    # Set number of rows and columns of the subplot
    nrows = ceil(np.sqrt(sample_number))
    ncols = ceil(sample_number / nrows)

    # Select a random sample of indices
    indices = sample(range(len(lines_list)), k=sample_number)

    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows))
    fig.suptitle('Examples of patches line(s) (in blue)', fontsize='30', fontweight='bold')

    for i, ax in enumerate(axes.flatten()):
        if i < sample_number:
            index = indices[i]  # select the index of the patch and line
            line = lines_list[index]
            patch = patches_list[index]

            height, width = np.shape(patch)  # Get the height and width of the image
            ax.set_xlim([0, width])  # Set the x-axis limits to match the width of the image
            ax.set_ylim([0, height])  # Set the y-axis limits to match the height of the image (note the inverted y-axis)
            ax.imshow(patch, extent=[0, height, 0, width], interpolation='nearest', cmap='copper')

            for segment in line:
                x_lim, y_lim = segment[0], segment[1]
                angle = calculate_angle(x_lim[0], y_lim[0], x_lim[1], y_lim[1])
                angle_deg = angle * 180 / np.pi
                # angle_norm = angle / (2 * np.pi)  # comment this out and add it to the caption if you want the normed angle
                ax.plot(x_lim, y_lim, color='blue', alpha=0.6, linewidth=5)
                ax.set_title('Angle: {:.3f} rad \n {:.3f}°'.format(angle, angle_deg), fontsize=20)

            if show_offset and (settings.label_offset_x != 0 or settings.label_offset_y != 0):
                # Create a rectangle patch that represent offset
                rect = patches.Rectangle((settings.label_offset_x - 0.5, settings.label_offset_y - 0.5),
                                         settings.patch_size_x - 2 * settings.label_offset_x,
                                         settings.patch_size_y - 2 * settings.label_offset_y,
                                         linewidth=2, edgecolor='fuchsia', facecolor='none')
                # Add the offset rectangle to the axes
                ax.add_patch(rect)
            ax.axis('off')
        else:
            fig.delaxes(ax)  # if there is no more patches but some axes are still to be filled, it deletes these axes and leaves a blank space

    save_plot(name)


def plot_samples(samples: List[torch.Tensor], lines: List[Any], title: str, file_name: str = None, confidences: List[Union[float, Tuple[float]]] = None,
                 show_offset: bool = True) -> None:
    """
    Plot a group of patches.

    :param samples: The list of patches to plot.
    :param lines: The lines associated
    :param title: The title of the plot.
    :param file_name: The file name of the plot if saved.
    :param confidences: The list of confidence score for the prediction of each sample. If it's a tuple then we assume
     it's (mean, std, entropy).
    :param show_offset: If True draws the offset rectangle (ignored if both offset x and y are 0)
    """
    plot_length = ceil(sqrt(len(samples)))

    if plot_length <= 1:
        return  # FIXME: deal with 1 or 0 sample

    # Create subplots
    fig, axs = plt.subplots(nrows=plot_length, ncols=plot_length, figsize=(plot_length * 2, plot_length * 2 + 1))

    for i, s in enumerate(samples):
        ax = axs[i // plot_length, i % plot_length]
        ax.imshow(s.reshape(settings.patch_size_x, settings.patch_size_y), interpolation='nearest', cmap='copper')

        if confidences:
            # If it's a tuple we assume it is: mean, std, entropy
            if isinstance(confidences[i], tuple):
                mean, std, entropy = confidences[i]
                ax.title.set_text(f'{mean:.2} (std {std:.2})\nEntropy:{entropy:.2}')
            # If it's not a tuple, we assume it is a float representing the confidence score
            else:
                ax.title.set_text(f'{confidences[i]:.2%}')

        if show_offset and (settings.label_offset_x != 0 or settings.label_offset_y != 0):
            # Create a rectangle patch that represent offset
            rect = patches.Rectangle((settings.label_offset_x - 0.5, settings.label_offset_y - 0.5),
                                     settings.patch_size_x - 2 * settings.label_offset_x,
                                     settings.patch_size_y - 2 * settings.label_offset_y,
                                     linewidth=2, edgecolor='fuchsia', facecolor='none')

            # Add the offset rectangle to the axes
            ax.add_patch(rect)

        ax.axis('off')

    fig.suptitle(title)
    plt.show()
    if file_name is not None:
        save_plot(f'samples_{file_name}')


class TextHandler(HandlerBase):
    """
    Custom legend handler for text field.
    From: https://stackoverflow.com/a/47382270/2666094
    """

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        h = copy(orig_handle)
        h.set_position((width / 2., height / 2.))
        h.set_transform(trans)
        h.set_ha("center")
        h.set_va("center")
        fp = orig_handle.get_font_properties().copy()
        fp.set_size(fontsize)
        h.set_font_properties(fp)
        return [h]


def plot_patch_test(patches: torch.Tensor, sample_number: int, angles_list: List, predicted_angle: Any, name: str = 'patch_sample') -> None:
    """
    Plot randomly sampled patches grouped by class to see if the network works well

    :param name: File name
    :param patches: The patches list to sample from.
    :param sample_number: The number of patches to sample
    :param angles_list: List of angles of lines if required
    :param predicted_angle: If given, will add the predicted angles value to the plot
    """
    # Check if sample number is not greater than the amount of available data
    n, p = patches.shape
    if (sample_number is not None) and (sample_number < n):
        logger.warning(f'{n} diagrams available but number of sampled diagram was set to {sample_number}. Only {n} will be sampled')
        n = sample_number

    nrows = ceil(np.sqrt(sample_number))
    ncols = ceil(sample_number / nrows)
    # Select a random sample of indices
    indices = sample(range(n), k=sample_number)
    # Create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    # fig.suptitle('Lines are in blue', fontsize='xx-large', fontweight='bold')
    figsize = (4 * ncols, 4 * nrows)
    plt.tight_layout()

    for i, ax in enumerate(axes.flatten()):
        if i < sample_number:
            height, width = settings.patch_size_x, settings.patch_size_y  # Get the height and width of the image

            index = indices[i]
            image = np.reshape(patches[index, :], (height, width))

            ax.set_xlim([0, width])  # Set the x-axis limits to match the width of the image
            ax.set_ylim([0, height])  # Set the y-axis limits to match the height of the image (note the inverted y-axis)

            ax.imshow(image, extent=[0, height, 0, width], interpolation='nearest', cmap='copper')

            normalized_angle = float(angles_list[index])
            angle = normalized_angle*(2*np.pi)
            pred_angle = predicted_angle[index][0]
            angle_degree = angle * 180 / np.pi

            title = 'Angle: {:.3f} | {:.3f}° \n Normalized value: {:.3f} \n Predicted value: {:.3f}'.format(angle, angle_degree, normalized_angle, pred_angle)
            ax.set_title(title, fontsize=20)

            ax.axis('off')
        else:
            fig.delaxes(ax)  # if there is no more patches but some axes are still to be filled, it deletes these axes and leaves a blank space

    save_plot(name)
