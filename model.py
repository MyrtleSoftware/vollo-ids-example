import torch

from torch import nn

from load import load

import numpy as np

from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2)

        self.dense = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        # X: [B, 90, 1]

        x = x.transpose(1, 2)  #  [B,  1, 90]
        x = self.conv(x)  #       [B, 64, 21]

        x = x.permute(2, 0, 1)  # [21, B, 64]
        x, _ = self.lstm(x)  #    [21, B, 64]

        # Take last element in the "time" dimension
        x = x[-1, :, :]  #   [B, 64]
        x = self.dense(x)  # [B, 1]

        return x
