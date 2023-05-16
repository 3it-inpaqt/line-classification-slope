import os
from typing import Any, Dict, Iterable, List, Tuple, Union
from dataclasses import asdict, is_dataclass
from copy import copy

from math import ceil
import numpy as np
import torch

from utils.settings import settings


def list_files(directory: str) -> List[str]:
    """
    List all files in a directory and returns a list with all files names
    :param directory: path to the directory
    :return: files list
    """
    os.chdir(directory)
    files = []
    for filename in os.listdir():
        if os.path.isfile(os.path.join(os.getcwd(), filename)):
            files.append(filename)
    return files


def create_directory(path: str) -> Any:
    """
    Create a new directory at the specified path
    :param path: Where the directory should be created
    """
    try:
        os.mkdir(path)
        print(f"Directory created at {path}")
    except FileExistsError:
        print(f"Directory already exists at {path}")


def create_txt_file(path: str, filename: str) -> Any:
    """
    Create a file given its name (with extension) and the path where to create it
    :param path: where to place the file
    :param filename: name of the file
    :return:
    """
    # check if directory path exists and is writable
    if not os.path.isdir(path) or not os.access(path, os.W_OK):
        raise OSError(
            f"Cannot create file '{filename}' in directory '{path}': directory does not exist or is not writable")

    # create full path of file
    file_path = os.path.join(path, filename)

    # create new file
    with open(file_path, 'w') as f:
        pass  # do nothing, just create empty file
    print(f"File '{filename}' in directory '{path}' was successfully created ")


def save_list_to_file(list1: list, path: str, list2=None) -> Any:
    """
    Save list to a file
    :param list1: list containing some values
    :param list2: second list, optional
    :param path: full path of the file (includes name and extension)
    :return:
    """
    # open file in write mode
    with open(path, 'w') as fp:
        if list2 is not None:
            for item1, item2 in enumerate(list1, list2):
                # write each item on a new line
                fp.write(f'{item1},{item2}\n')
        else:
            for item1 in list1:
                # write each item on a new line
                fp.write(f'{item1}\n')
        print('Done')


def clip(n, smallest, largest):
    """ Shortcut to clip a value between 2 others """
    return max(smallest, min(n, largest))


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
    Dictionary containing a list of dictionary for the initial and final coordinates of a line points
    :param label_dic:
    :return: tuple of coordinate x1, y1, x2, y2
    """
    line = label_dic['line']
    x1, y1 = line[0]['x'], line[0]['y']
    x2, y2 = line[1]['x'], line[1]['y']

    return x1, y1, x2, y2