import torch

# from model import Net

import torch.nn as nn

import vollo_torch

from vollo_torch.nn import PaddedConv1d, LSTM


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            PaddedConv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            PaddedConv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
        )

        self.lstm = LSTM(input_size=64, hidden_size=128, num_layers=2)

        self.dense = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            # nn.Sigmoid(),
        )

    # def forward(self, x):

    #     assert x.dim() in (2, 3)

    #     x = x.transpose(1, 2)  #  [B,  1, 90]
    #     x = self.conv(x)  #       [B, 64, 21]

    #     if x.dim() == 3:
    #         x = x.permute(2, 0, 1)  # [21, B, 64]
    #     else:
    #         x = x.transpose(0, 1)  # [21, 64]

    #     x = self.lstm(x)  #    [21, B, 64]

    #     # Take last element in the "time" dimension
    #     x = x[-1, :, :]  #   [B, 64]
    #     x = self.dense(x)  # [B, 1]

    #     return x

    def forward(self, x):

        x = self.lstm(x)  #    [21, B, 64]

        # print(x.shape)

        return x


model = Net()

model.eval()


# Dummy input for tracing
input = torch.randn(1, 1, 64)

model(input)

# exit(0)

# Trace the model's execution to annotate it with activation shapes
(model, expected_output) = vollo_torch.fx.prepare_shape(model, input)

# batch_size = 1
# sequence_length = 5
# input = torch.randn(batch_size, in_channels, sequence_length)
# (model, expected_output) = vollo_torch.fx.prepare_shape(model, input)
# nnir = vollo_torch.fx.nnir.to_nnir(model)

# # Provide the streaming transform with index of the sequence axis


nnir = vollo_torch.fx.nnir.to_nnir(model)

# (nnir, output_axis) = nnir.streaming_transform(1)

import vollo_compiler

config = vollo_compiler.Config.ia_420f_c6b32()
program = nnir.to_program(config)

print(program.metrics())
