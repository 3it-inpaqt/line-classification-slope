from linegeneration.generatelines import create_image_set
from plot.linesvizualisation import create_multiplots
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Generates images
    N = 18
    n = 25
    batch, angle_list = create_image_set(n, N)
    fig, axes = create_multiplots(batch, angle_list)
    plt.tight_layout()
    plt.show()
