import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional classifier neural network.
    """
    def __init__(self):
        super(CNN, self).__init__()  # ensures the initialization of the parent nn.Module class is called before defining any custom layers or parameters.

        # Device will determine whether to run the training on GPU or CPU.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Layers
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 12, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(768, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
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
