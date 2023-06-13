import torch
import torch.nn as nn
from torch import optim
from typing import Any, Tuple


class CNN(nn.Module):
    """
    Convolutional classifier neural network.
    """
    def __init__(self, batch_size: int, learning_rate: float, epochs: int):
        # Device will determine whether to run the training on GPU or CPU.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Parameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Binary Cross Entropy including sigmoid layer
        self._criterion = nn.BCEWithLogitsLoss()
        self._optimizer = optim.Adam(lr=learning_rate)

        # Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 4 * 4, 1)

    def forward(self, x):
        """
        Define the forward logic
        :param x: One input of the dataset
        :return: Output of the network
        """
        out1 = self.conv1(x)
        out2 = self.pool1(out1)
        out3 = self.conv2(out2)
        out4 = self.pool2(out3)

        out = out4.reshape(out4.size(0), -1)
        out = self.fc(out)

        return out

    def training_step(self, inputs: Any, labels: Any) -> float:
        """
        Define the logic for one training step.

        :param inputs: The input from the training dataset, could be a batch or an item
        :param labels: The label of the item or the batch
        :return: The loss value
        """
        # Zero the parameter gradients
        self._optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = self(inputs)
        loss = self._criterion(outputs, labels.float())
        loss.backward()
        self._optimizer.step()

        return loss.item()

