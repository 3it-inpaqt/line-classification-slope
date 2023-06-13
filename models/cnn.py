import torch
import torch.nn as nn
from torch import optim
from typing import Tuple


class CNN(nn.Module):
    """
    Convolutional classifier neural network.
    """
    def __init__(self, input_shape: Tuple[int, int]):
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 4 * 4, 1)
