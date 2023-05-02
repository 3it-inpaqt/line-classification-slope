import torch.nn as nn


def AngleNet(N):
    model = nn.Sequential(
        nn.Linear(N*N, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )
    return model
