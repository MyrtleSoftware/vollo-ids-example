import torch

from torch import nn

from load import load

import numpy as np

from tqdm import tqdm


class ResFFN(nn.Module):
    def __init__(self, n: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(n, 2 * n),
            nn.ReLU(),
            nn.Linear(2 * n, n),
        )

    def forward(self, x):
        return x + self.net(x)


class Net(nn.Module):
    def __init__(self, input_size=95, hid: int = 128):

        super().__init__()

        self.pre = nn.Sequential(
            nn.Linear(input_size, hid),
            ResFFN(hid),
        )

        self.lstm = nn.LSTM(
            input_size=hid, hidden_size=2 * hid, num_layers=1, batch_first=True
        )

        self.post = nn.Sequential(
            # ResFFN(hid),
            nn.Linear(2 * hid, 1),
            nn.Dropout(0.2),
            nn.Sigmoid(),
        )

    def forward(self, x):

        # print(x.shape)

        # x = self.bn(x)

        # X: [B, H, 1]

        x = self.pre(x)
        x, _ = self.lstm(x)
        x = x[:, :, :]  # Last time step
        x = self.post(x)

        # print(x.shape)

        return x
