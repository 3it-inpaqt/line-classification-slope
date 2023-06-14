import torch.nn as nn

# I don't know if it is useful for the feed-forward on a pretrained network, but it works just fine
# so if you have any suggestions feel free to share them


def AngleNet(N):
    model = nn.Sequential(
                nn.Linear(N, 24),  # change N to N*N if you use synthetic data
                nn.LeakyReLU(),
                nn.Linear(24, 6),
                nn.LeakyReLU(),
                nn.Linear(6, 1),
        )
    return model
