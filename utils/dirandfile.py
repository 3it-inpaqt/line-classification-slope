import os
from typing import List, Any


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
