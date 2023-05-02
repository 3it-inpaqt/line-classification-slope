import torch.nn as nn


class AngleNet(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        self.model = nn.Sequential(
            nn.Linear(self.N*self.N, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )
