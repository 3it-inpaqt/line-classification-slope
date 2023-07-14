import matplotlib.pyplot as plt
import os
import torch
from typing import Any, List, Optional

from utils.logger import logger
from utils.misc import dec_to_sci
from utils.settings import settings


def save_model(model, filename: str, directory_path: str, loss_history: List[Any], best_loss: float, accuracy: float, standard_deviation: float) -> None:
    """
    Save a machine learning model in a .pt file.
    :param model: Pytorch model to save, by default simply 'model'
    :param filename: String, if the file already exist, an index is added
    :param directory_path: Path of the master directory to save all the files (models and plots)
    :param loss_history: Associated list of the loss evolution
    :param best_loss: Best loss of the model
    :param accuracy: Accuracy of the model
    :param standard_deviation: Best value of standard deviation
    :return: save model to a directory called 'saved', feel free to change the name if you want
    """
    # Define path to 'saved' folder
    run_type = settings.model_type
    if run_type == "FF":
        path = f"./{directory_path}/{settings.research_group}/regression/{settings.loss_fn}"
    elif run_type == "CNN":
        path = f"./{directory_path}/{settings.research_group}/convolution/{settings.loss_fn}"
    else:
        while run_type != "ff" and run_type != "cnn":
            logger.warning(f'Run type invalid.')
            run_type = input('Please select a proper run type (cnn or ff): ')

    # Add extension if not provided
    model_filename = filename + '.pt'  # for tensor file
    plot_filename = filename + '.png'  # for plot file

    # Check if directory exists, create if it does not
    if os.path.exists(path):
        print(f'Output directory {path} exists, saving file')
    else:
        os.makedirs(path)
        print(f'Output directory created: {path}')

    # Append index to filename to avoid overwriting previously saved models
    index = 0
    indexed_model_name = model_filename  # avoid repetition in the name, as filename is being written over and over
    indexed_plot_name = plot_filename
    while os.path.exists(os.path.join(path, indexed_model_name)):
        index += 1
        indexed_model_name = f"{model_filename[:-3]}_{index}.pt"  # makes sure extension is not in the middle of the name
        indexed_plot_name = f"{plot_filename[:-4]}_{index}.png"

    # Plot accuracy
    fig, ax = plt.subplots()
    ax_title = f'Training on the experimental patches \n Learning rate: {settings.learning_rate} | Epochs: {settings.n_epochs} | Batch: {settings.batch_size}' #  | Threshold: {settings.threshold_loss}°

    ax.set_title(ax_title)
    # print("Loss: %.4f" % best_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.plot(loss_history)

    # Add a text box to the plot
    textstr = '\n'.join((
        r'$Best Loss = {{{loss}}}$'.format(loss=dec_to_sci(best_loss), ),
        r'$\sigma = {{{deviation}}} $'.format(deviation=dec_to_sci(standard_deviation), ),
        r'$Accuracy = {{{acc}}}$'.format(acc=accuracy, ),
        f'{settings.n_hidden_layers} hidden layers',
        f'{settings.loss_fn}'
    ))

    if settings.loss_fn == 'SmoothL1Loss':
        textstr = '\n'.join((textstr,
                             r'$\beta = {{{beta}}}$'.format(beta=settings.beta, )
                             ))

    elif settings.loss_fn == 'HarmonicFunctionLoss':
        textstr = '\n'.join((textstr,
                             r'$n = {{{num_harmonic}}}$'.format(num_harmonic=settings.num_harmonics, )
                             ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.9, 0.9, textstr, transform=ax.transAxes, fontsize=14, ha='right', va='top', bbox=props)

    # Save the figure to a file
    os.chdir(path)
    plt.savefig(f'{model_filename}.png')
    logger.info(f"Plot saved to {indexed_plot_name}")
    plt.show()

    # Save the model to a file
    # filepath = os.path.join(path, indexed_model_name)
    torch.save(model.state_dict(), indexed_model_name)
    logger.info(f"Model saved to {indexed_model_name}")
