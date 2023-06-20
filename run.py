from linegeneration.generate_lines import create_image_set
from plot.lines_visualisation import create_multiplots
import matplotlib.pyplot as plt
import torch

from models.model import AngleNet
from utils.angle_operations import normalize_angle
from utils.settings import settings
from utils.statistics import calculate_std_dev

if __name__ == '__main__':

    run_type = settings.model_type

    if run_type == 'FF':
        from models.run_regression import main
        main()

    elif run_type == 'CNN':
        from models.run_cnn import main
        main()

    # image_set_test, angles_test = create_image_set(n, N, 0.9, aa=True)
    # fig, axes = create_multiplots(image_set_test, angles_test, number_sample=n)
    # plt.show()

