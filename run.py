from linegeneration.generatelines import create_image_set
from plot.linesvizualisation import create_multiplots
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Generates images
    N = 18
    n = 25
    image_set, angles = create_image_set(n, N)
    fig, axes = create_multiplots(image_set, angles)

    plt.tight_layout()
    plt.show()
