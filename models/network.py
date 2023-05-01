import torch
import torch.nn as nn


class AngleNet(nn.Module):
    def __init__(self):
        super(AngleNet, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.hidden_layers = nn.Sequential(
            nn.Linear(in_features=128 * 2 * 2, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        # Flatten output from convolutional layers
        x = x.view(-1, 128 * 2 * 2)
        x = self.hidden_layers(x)

        return x
