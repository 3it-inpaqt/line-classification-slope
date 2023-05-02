import os
import torch


def save_model(model, filename='model'):
    """

    :param model: Pytorch model to save
    :param filename: String, by default the model is saved as 'model', if the file already exist, an index is added
    :return: save model to a directory called 'saved', feel free to change the name if you want
    """
    # Define path to 'saved' folder
    path = '.\saved'

    # Add extension if not provided
    if not ('.pt' in filename):
        filename += '.pt'

    # Check if directory exists, create if it does not
    if os.path.exists(path):
        print(f'Output directory {path} exists, saving file')
    else:
        os.makedirs(path)
        print(f'Output directory created: {path}')

    # Append index to filename to avoid overwriting previously saved models
    index = 0
    indexed_name = filename  # avoid repetition in the name, as filename is being written over and over
    while os.path.exists(os.path.join(path, indexed_name)):
        print(os.path.join(path, indexed_name))
        index += 1
        indexed_name = f"{filename[:-3]}_{index}.pt"  # makes sure extension is not in the middle of the name

    # Save the model to a file
    filepath = os.path.join(path, indexed_name)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
