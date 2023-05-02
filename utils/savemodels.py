import os
import torch

os.umask(0)  # unmask user to let python write files and directories


def save_model(model, filename='model'):
    """

    :param model: Pytorch model to save
    :param filename: String, by default the model is saved as 'model', if the file already exist, an index is added
    :return: save model to a directory called 'saved', feel free to change the name if you want
    """
    # Define path to 'saved' folder
    path = '.\saved'
    print(path)

    # Check if directory exists, create if it does not
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Output directory created: {path}')
    else:
        print(f'Output directory {path} exists, saving file')

    # Append index to filename to avoid overwriting previously saved models
    index = 0
    while os.path.exists(os.path.join(path, filename)):
        index += 1
        filename = f"{filename}_{index}.pt"

    # Save the model to a file
    filepath = os.path.join(path, filename)
    print(filepath)
    with open(filepath, 'w') as f:
        torch.save(model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
