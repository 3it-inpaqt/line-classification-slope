import torch
import torch.nn as nn
from numpy import sin

from utils.settings import settings


# Angle difference loss
class AngleDifferenceLoss(nn.Module):
    """
    This loss function calculates the difference between the angles of the predicted and actual slopes. It can help the
    network focus on minimizing the angle difference between the predicted and actual slopes.
    """
    def __init__(self):
        super(AngleDifferenceLoss, self).__init__()

    @staticmethod
    def forward(y_pred, y_batch):
        angle_diff = torch.abs(torch.atan(y_pred) - torch.atan(y_batch))
        loss = torch.mean(angle_diff)
        return loss


#  Weighted SmoothL1Loss
class WeightedSmoothL1Loss(nn.Module):
    """
    This loss function combines the SmoothL1Loss with a weight that depends on the slope value. It can help the network
    focus more on specific slope ranges.
    """
    def __init__(self, beta):
        super(WeightedSmoothL1Loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=beta)

    def forward(self, y_pred, y_batch):
        loss = self.smooth_l1_loss(y_pred, y_batch)
        weights = torch.abs(y_batch * 2 * torch.pi)
        weighted_loss = torch.mean(loss * weights)
        return weighted_loss


# Harmonic loss
class HarmonicMeanLoss(nn.Module):
    """
    This loss function calculates the harmonic mean of the SmoothL1Loss and the Angle Difference Loss. It can help the
    network balance between minimizing the angle difference and the SmoothL1Loss.
    """
    def __init__(self):
        super(HarmonicMeanLoss, self).__init__()
        self.angle_difference_loss = AngleDifferenceLoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, y_pred, y_batch):
        angle_diff_loss = self.angle_difference_loss(y_pred, y_batch)
        smooth_l1_loss = self.smooth_l1_loss(y_pred, y_batch)
        harmonic_loss = 2 * (angle_diff_loss * smooth_l1_loss) / (angle_diff_loss + smooth_l1_loss)
        return harmonic_loss


class HarmonicFunctionLoss(nn.Module):
    """
    This custom loss function calculates the difference between the harmonic functions of the predicted and actual
    slopes. The num_harmonics parameter determines the number of harmonics to consider in the calculation. You can
    experiment with different numbers of harmonics and see how it affects the performance of the network.
    """
    def __init__(self, num_harmonics=5):
        super(HarmonicFunctionLoss, self).__init__()
        self.num_harmonics = num_harmonics

    def forward(self, y_pred, y_batch):
        harmonics = torch.stack([torch.sin((n + 1) * y_pred) for n in range(self.num_harmonics)])
        target_harmonics = torch.stack([torch.sin((n + 1) * y_batch) for n in range(self.num_harmonics)])
        loss = torch.mean(torch.abs(harmonics - target_harmonics))
        return loss


# All the losses to use
loss_fn_dic = {'SmoothL1Loss': nn.SmoothL1Loss(),
               'MSE': nn.MSELoss(),
               'MAE': nn.L1Loss(),
               'AngleDiff': AngleDifferenceLoss(),
               'WeightedSmoothL1': WeightedSmoothL1Loss(beta=settings.beta),
               'HarmonicMeanLoss': HarmonicMeanLoss(),
               'HarmonicFunctionLoss': HarmonicFunctionLoss(num_harmonics=settings.num_harmonics)}
