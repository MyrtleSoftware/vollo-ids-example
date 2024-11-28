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


class DataLoader:
    def __init__(self, bath_size=1024, device="cuda"):
        self.X, self.Y = load()
        self.batch_size = bath_size
        self.device = device

    def iter(self, split, drop_last=True):

        x = self.X[split].to_numpy()
        y = self.Y[split].to_numpy()

        # Concat for shuffling
        xy = np.concatenate([x, y], axis=1)

        # Shuffle the rows
        np.random.shuffle(xy)

        n = x.shape[1]

        x, y = xy[:, :n], xy[:, n:]

        # Drop last elements so that the batch size divides evenly
        n = x.shape[0] // self.batch_size * self.batch_size

        x, x_rem = x[:n], x[n:]
        y, y_rem = y[:n], y[n:]

        x = x.reshape(-1, self.batch_size, x.shape[1])
        y = y.reshape(-1, self.batch_size, y.shape[1])

        kwargs = {"dtype": torch.float32, "device": self.device}

        for xx, yy in zip(x, y):
            yield torch.tensor(xx, **kwargs), torch.tensor(yy, **kwargs)

        if not drop_last:
            yield torch.tensor(x_rem, **kwargs), torch.tensor(y_rem, **kwargs)


# =================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =================

model = Net().to(device)
print("Parameters:", sum(p.numel() for p in model.parameters()))
print(model)

# =================

optimizer = torch.optim.Adam(model.parameters())

# =================

loader = DataLoader(device=device)


def eval(iter):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for x, y in iter:

        pred = model(x[:, :, None])

        pred = pred > 0.5
        target = y[:, :1] > 0.5

        tp += (pred & target).sum().item()
        fp += (pred & ~target).sum().item()
        tn += (~pred & ~target).sum().item()
        fn += (~pred & target).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    t = tp + tn + fp + fn

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "confusion_matrix": [[tp / t, fp / t], [fn / t, tn / t]],
    }


for _ in range(1):
    for x, y in tqdm(loader.iter("train"), leave=False):

        optimizer.zero_grad()

        logits = model(x[:, :, None])
        # The first column of y is attack/!attack
        loss = torch.nn.functional.binary_cross_entropy(logits, y[:, :1])

        loss.backward()
        optimizer.step()

    print("train set", eval(loader.iter("train", drop_last=False)))

    print("dev set", eval(loader.iter("dev", drop_last=False)))

print("test set", eval(loader.iter("test", drop_last=False)))
