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

        # self.bn = nn.BatchNorm1d(input_size)

        self.pre = nn.Sequential(
            nn.Linear(input_size, hid),
            ResFFN(hid),
            ResFFN(hid),
            nn.Linear(hid, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        # x = self.bn(x)

        # X: [B, H, 1]

        x = self.pre(x.squeeze(2))

        # print(x.shape)

        return x
