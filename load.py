import pandas as pd
import os
import torch
import numpy as np

_ATTACKS = {
    "Normal": 0,
    "MITM": 1,
    "Uploading": 2,
    "Ransomware": 3,
    "SQL_injection": 4,
    "DDoS_HTTP": 5,
    "DDoS_TCP": 6,
    "Password": 7,
    "Port_Scanning": 8,
    "Vulnerability_scanner": 9,
    "Backdoor": 10,
    "XSS": 11,
    "Fingerprinting": 12,
    "DDoS_UDP": 13,
    "DDoS_ICMP": 14,
}


def _load(csv_file: str, cache_file: str):

    if os.path.exists(cache_file):
        df = pd.read_pickle(cache_file)
    else:
        df = pd.read_csv(csv_file, low_memory=False)

    df["Attack_type"] = df["Attack_type"].map(_ATTACKS)

    x = df.drop(columns=["Attack_label", "Attack_type"])
    y = df.loc[:, ["Attack_label", "Attack_type"]]

    return x, y


def load(dir: str = "build/"):

    keys = ["train", "dev", "test"]

    x = {}
    y = {}

    for k in keys:
        x[k], y[k] = _load(os.path.join(dir, f"{k}.csv"), os.path.join(dir, f"{k}.pkl"))

    return x, y


class DataLoader:
    def __init__(self, bath_size=1024, device="cuda"):

        self.X, self.Y = load()

        self.batch_size = bath_size
        self.device = device

        # Normalize features to zero mean and unit variance

        mean = self.X["train"].mean()
        var = self.X["train"].var()

        for k in self.X:
            self.X[k] = (self.X[k] - mean) / (var + 1e-6) ** 0.5

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


if __name__ == "__main__":
    load()
