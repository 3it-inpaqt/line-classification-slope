from math import prod
from torch import sin, cos, mean, std_mean, abs, pi
import torch.nn as nn
from torch import Tensor, gt, min
from typing import List

from models.feed_forward import FeedForward
from utils.settings import settings


def AngleNet(input_shape: int, n_hidden_layers: int):
    """

    :param input_shape: Shape of the input (N*N)
    :param n_hidden_layers: Number of hidden layers
    :return:
    """
    if n_hidden_layers == 1:
        model = nn.Sequential(
            nn.Linear(input_shape, 24),
            nn.LeakyReLU(),
            nn.Linear(24, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    elif n_hidden_layers == 2:
        model = nn.Sequential(
                nn.Linear(input_shape, 24),
                nn.LeakyReLU(),
                nn.Linear(24, 6),
                nn.LeakyReLU(),
                nn.Linear(6, 3),
                nn.ReLU(),
                nn.Linear(3, 1)
            )
    return model


def find_loss(y_pred: Tensor, y: Tensor, criterion, threshold=0.):
    """
    Method to optimize the loss to avoid huge gap between values predicted and expected. Especially for angles close to
    180° and 0°.
    :param y_pred:
    :param y:
    :param criterion:
    :param threshold:
    :return:
    """
    new_y_pred = y_pred.clone()

    # Apply the operation to elements greater than the threshold
    mask = gt(y_pred, threshold)
    new_y_pred[mask] = y_pred[mask] - 0.5

    # Calculate the loss between y and y_pred, and y and new_y_pred
    loss1 = criterion(y_pred, y)
    loss2 = criterion(new_y_pred, y)
    # Return the smaller loss
    return min(loss1, loss2)


def harmonic_loss(y_pred, y):
    """
    Calculate loss using harmonic function
    :param y_pred:
    :param y:
    :return:
    """
    return mean((sin(y_pred) - sin(y))**2)
