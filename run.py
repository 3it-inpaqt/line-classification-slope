from linegeneration.generatelines import create_batch
from utils.angleoperations import calculate_angle
from plot.linesvizualisation import create_multiplots
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Generates images
    size = (18, 18)
    n = 40
    batch = create_batch(n, size)
    fig, axes = create_multiplots(batch)

    plt.show()
