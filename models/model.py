import torch.nn as nn
from torch import Tensor, gt, min
# I don't know if it is useful for the feed-forward on a pretrained network, but it works just fine
# so if you have any suggestions feel free to share them

loss_fn_dic = {'SmoothL1Loss': nn.SmoothL1Loss(), 'MSE': nn.MSELoss(), 'MAE': nn.L1Loss()}


def AngleNet(input_size, n_hidden_layers):
    if n_hidden_layers == 1:
        model = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.LeakyReLU(),
            nn.Linear(24, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )
    elif n_hidden_layers == 2:
        model = nn.Sequential(
                nn.Linear(input_size, 24),
                nn.LeakyReLU(),
                nn.Linear(24, 12),
                nn.LeakyReLU(),
                nn.Linear(12, 6),
                nn.ReLU(),
                nn.Linear(6, 1)
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
