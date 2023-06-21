import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional classifier neural network.
    """
    def __init__(self, kernel_size_conv=4):
        super(CNN, self).__init__()  # ensures the initialization of the parent nn.Module class is called before defining any custom layers or parameters.

        # Device will determine whether to run the training on GPU or CPU.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Layers
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=kernel_size_conv, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(6, 12, kernel_size=kernel_size_conv, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Flatten(),
            # nn.MaxPool2d(kernel_size=kernel_size_maxpool, stride=2),
            nn.Linear(12 * (kernel_size_conv ** 2) * (kernel_size_conv ** 2), 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1)
            )

    def forward(self, x):
        """
        Define the forward logic
        :param x: One input of the dataset
        :return: Output of the network
        """
        out = self.layers(x)
        return out
