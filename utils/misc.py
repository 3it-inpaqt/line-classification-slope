from dataclasses import asdict, is_dataclass
from copy import copy
from math import ceil
import numpy as np
import os
import random
import torch
from typing import Any, Dict, List, Tuple, Union

from utils.settings import settings
from utils.logger import logger


# -- os process (files, directory, etc.) -- #

def list_files(directory: str, extension: str = None) -> List[str]:
    """
    List all files in a directory and returns a list with all files names
    :param directory: Path to the directory
    :param extension: Optional, extension of files to list
    :return: files list
    """
    os.chdir(directory)
    files = []

    for filename in os.listdir():
        if os.path.isfile(os.path.join(os.getcwd(), filename)):
            if extension is None or filename.endswith(extension):
                files.append(filename)

    return files


def create_directory(path: str) -> Any:
    """
    Create a new directory at the specified path
    :param path: Where the directory should be created
    """
    try:
        os.mkdir(path)
        logger.info(f"Directory created at {path}")
    except FileExistsError:
        logger.warning(f"Directory already exists at {path}")


def create_txt_file(path: str, filename: str) -> Any:
    """
    Create a file given its name (with extension) and the path where to create it
    :param path: where to place the file
    :param filename: name of the file
    :return:
    """
    # Check if directory path exists and is writable
    if not os.path.isdir(path) or not os.access(path, os.W_OK):
        raise OSError(
            f"Cannot create file '{filename}' in directory '{path}': directory does not exist or is not writable")

    # Create full path of file
    file_path = os.path.join(path, filename)

    # Create new file
    with open(file_path, 'w') as f:
        pass  # do nothing, just create empty file

    logger.info(f"File '{filename}' in directory '{path}' was successfully created ")


def save_list_to_file(list1: list, path: str, list2=None) -> Any:
    """
    Save list to a file
    :param list1: list containing some values
    :param list2: second list, optional
    :param path: full path of the file (includes name and extension)
    :return:
    """
    # Open file in write mode
    with open(path, 'w') as fp:
        if list2 is not None:
            for item1, item2 in enumerate(list1, list2):
                # write each item on a new line
                fp.write(f'{item1},{item2}\n')
        else:
            for item1 in list1:
                # write each item on a new line
                fp.write(f'{item1}\n')
    logger.info('List(s) saved to file')


def load_list_from_file(filepath: str) -> List[Any]:
    """
    Load a list from a text file
    :param filepath:
    :return:
    """
    # Prepare the list to store the data
    list_from_file = []

    with open(filepath) as f:
        for line in f:
            list_from_file.append(line.strip())

    return list_from_file


# -- YAML and JSON --#

def yaml_preprocess(item: Any) -> Union[str, int, float, List, Dict]:
    """
    Convert complex object to datatype accepted by yaml format.

    :param item: The item to process.
    :return: The converted item.
    """
    # FIXME: detect recursive structures

    # Convert Numpy accurate float representation to standard python float
    if isinstance(item, np.float_):
        return float(item)

    # If a primitive type know by yalm, then everything is good,
    if isinstance(item, str) or isinstance(item, int) or isinstance(item, float) or isinstance(item, bool):
        return item

    # If dataclass use dictionary conversion
    if is_dataclass(item) and not isinstance(item, type):
        item = asdict(item)

    # If it's a dictionary, process the values
    if isinstance(item, dict):
        item = copy(item)
        for name, value in item.items():
            item[name] = yaml_preprocess(value)  # TODO Process name too?

        return item

    try:
        # Try to convert to a list, if not possible throws error and convert it to string
        item = list(item)
        item = copy(item)

        for i in range(len(item)):
            item[i] = yaml_preprocess(item[i])

        return item

    except TypeError:
        # Not iterable, then convert to string
        return str(item)


def format_string(input_string):
    """
    Format the Dataset Name from json file to match the settings research group name
    :param input_string: Typically "QDSD - Michel Pioro Ladriere"
    :return: formatted string, no more spaces, small caps only and no useless characters
    """
    # Replace spaces with underscores
    formatted_string = input_string.replace(' ', '_')

    # Remove first 7 characters (QDSD - )
    formatted_string = formatted_string[7:]

    # Convert to lowercase
    formatted_string = formatted_string.lower()

    return formatted_string


def convert_coordinate_dic(label_dic):
    """
    Dictionary containing a list of dictionary for the initial and final coordinates of a line points. Initial goal was
    to collect from the json file the information on the line but this is already handled by the load_diagram function.
    :param label_dic:
    :return: tuple of coordinate x1, y1, x2, y2
    """
    line = label_dic['line']
    x1, y1 = line[0]['x'], line[0]['y']
    x2, y2 = line[1]['x'], line[1]['y']

    return x1, y1, x2, y2


# -- Randomization and resampling -- #

def random_select_elements(list1: List[Any], list2: List[Any], num_elements: int = 1) -> tuple[Any, Any, int]:
    """
    Select two elements from two different list with same index randomly
    :param list1:
    :param list2:
    :param num_elements: number of elements to select
    :return:
    """
    # Randomly select indices without replacement
    index = random.randint(0, len(list1) - 1)

    # Use the selected index to get the corresponding elements from both lists
    selected_elements1 = list1[index]
    selected_elements2 = list2[index]

    return selected_elements1, selected_elements2, index


# -- Related to float and integer -- #
def isqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def dec_to_sci(number):
    """
    Convert float into scientific notation. Remove all trailing zeros automatically.
    Based on https://stackoverflow.com/a/6913576/19392385
    :param number:
    :return:
    """
    a = '%E' % number
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


def clip(n, smallest, largest):
    """ Shortcut to clip a value between 2 others """
    return max(smallest, min(n, largest))


# -- Tensor processing -- #

def get_nb_loader_workers(device: torch.device = None) -> int:
    """
    Estimate the number based on: the device > the user settings > hardware setup

    :param device: The torch device.
    :return: The number of data loader workers.
    """

    # Use the pyTorch data loader
    if device and device.type == 'cuda':
        # CUDA doesn't support multithreading for data loading
        nb_workers = 0
    elif settings.nb_loader_workers:
        # Use user setting if set (0 mean auto)
        nb_workers = settings.nb_loader_workers
    else:
        # Try to detect the number of available CPU
        # noinspection PyBroadException
        try:
            nb_workers = len(os.sched_getaffinity(0))
        except Exception:
            nb_workers = os.cpu_count()

        nb_workers = ceil(nb_workers / 2)  # The optimal number seems to be half of the cores

    return nb_workers


def enhance_contrast(tensor, threshold=0.3):
    """
    Enhance tensor contrast. If the value of the tensor is greater than 0.5 it gets multiplied by 1.5 and if below 0.5
    it gets multiplied by 0.6. Light gets lighter and dark gets darker.
    :param threshold:
    :param tensor:
    :return:
    """
    # Initialize a new tensor with the same dimensions as the input tensor
    enhanced_tensor = torch.zeros_like(tensor)

    # Apply contrast enhancement
    mask = tensor > threshold
    enhanced_tensor[mask] = tensor[mask] * 1.5
    enhanced_tensor[~mask] = tensor[~mask] * 0.6

    return enhanced_tensor


def renorm_array(input_table: Any) -> torch.Tensor:
    """
    Re-normalise a tensor with value between 0 and 1
    :param input_table: Tensor of size [N, N] or Array
    :return: Normalized tensor of size [1, N, N]
    """
    if torch.is_tensor(input_table):
        tensor = input_table.clone()
    else:
        tensor = torch.from_numpy(input_table)

    new_tensor = tensor.view(1, -1)
    new_tensor -= new_tensor.min(1, keepdim=True)[0]
    new_tensor /= new_tensor.max(1, keepdim=True)[0]
    new_tensor = new_tensor.view(tensor.size(0), tensor.size(1), tensor.size(2))

    return new_tensor.squeeze(1)


def resymmetrise_tensor(y_pred: torch.Tensor, threshold=0.):
    """
    Takes angle prediction array, and makes sure the angle are withing the range [0°, 180°[
    :param y_pred:
    :param threshold:
    :return:
    """
    new_y_pred = y_pred.clone()

    # Apply the operation to elements greater than the threshold
    mask = torch.gt(y_pred, threshold)
    new_y_pred[mask] = y_pred[mask] - 1

    # Return the smaller loss
    return new_y_pred
