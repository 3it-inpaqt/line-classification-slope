import pandas as pd
import os

from utils.settings import settings
from utils.logger import logger


def init_csv(loss, std_dev):
    """ Initialise CSV files for each loss function name and save them in a dedicated folder
    :param loss:
    :param std_dev:
    """
    # Define the folder name and create the folder if it doesn't exist
    folder = './saved/csv_files'
    os.makedirs(folder, exist_ok=True)

    # Define the file name and column headers
    filename = os.path.join(folder, f'{settings.loss_fn}.csv')
    fields = ['Learning Rate', 'Epochs', 'Batch Size', 'Hidden layers', 'Threshold loss']
    values = [settings.learning_rate, settings.n_epochs, settings.batch_size, settings.n_hidden_layers, settings.threshold_loss]
    if settings.loss_fn in ['SmoothL1Loss', 'HarmonicMeanLoss', 'WeightedSmoothL1']:
        fields.append('Beta')
        values.append(settings.beta)
    elif settings.loss_fn == 'HarmonicFunctionLoss':
        fields.append('Harmonic Number')
        values.append(settings.num_harmonics)

    fields.extend(['Loss', 'Standard Deviation'])
    values.extend([loss, std_dev])

    # Create a CSV file for each loss function name
    if not os.path.isfile(filename):
        # Create the CSV file and write the header
        df = pd.DataFrame(columns=fields)
        df.to_csv(filename, index=False)

    # Write the results to the CSV file
    # Open the CSV file and write the statistics to the appropriate sheet

    df = pd.DataFrame([values],
                      columns=fields)
    df.to_csv(filename, mode='a', header=False, index=False)

    logger.info('Settings and results saved in CSV file')
