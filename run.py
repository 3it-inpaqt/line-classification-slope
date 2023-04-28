from linegeneration.generatelines import generate_image
from utils.angleoperations import calculate_angle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Generates image
    size = (18, 18)
    image = generate_image(size)
    angle_radian, angle_degree = calculate_angle(image)

    # Generating figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.axis('off')
    ax.imshow(image, cmap='Greys')
    plt.tight_layout()
    plt.show()
    print('Angle in radian: ', angle_radian)
    print('Angle in degree: ', angle_degree)
