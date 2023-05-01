from linegeneration.generatelines import create_batch
from utils.angleoperations import calculate_angle
from plot.linesvizualisation import create_multiplots
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Generates images
    N = 18
    n = 25
    batch, angle_list = create_batch(n, N)
    fig, axes = create_multiplots(batch, angle_list)

    plt.show()
