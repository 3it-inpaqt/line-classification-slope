"""
Easy way to run the model on the data. Just change the parameters in settings.yaml, and then you can run this script to
train a model.
"""
from utils.logger import logger
from utils.settings import settings


if __name__ == '__main__':

    run_type = settings.model_type  # fetch the type of NN to use

    if run_type == 'FF':
        from models.run_regression import main
        main()

    elif run_type == 'CNN':
        from models.run_cnn import main
        main()

    elif run_type == 'EDGE-DETECT':
        # from models.edge_detection import main
        # main()
        logger.critical('Method not working, WONT-FIX!')
