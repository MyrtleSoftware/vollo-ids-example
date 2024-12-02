from torch import nn


class Net(nn.Module):
    def __init__(self, input_size, hid: int = 128):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hid,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )

        self.post = nn.Sequential(
            nn.Linear(hid, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, state=None):

        x, state = self.lstm(x, state)

        x = self.post(x)

        return x, state
