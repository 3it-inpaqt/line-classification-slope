import torch.nn as nn

# I don't know if it is useful for the feed-forward on a pretrained network, but it works just fine
# so if you have any suggestions feel free to share them


def AngleNet(N):
    model = nn.Sequential(
        nn.Linear(N*N, 24),
        nn.LeakyReLU(),
        nn.Linear(24, 12),
        nn.LeakyReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )
    return model
