import pandas as pd
import os
import torch
import numpy as np
import re


def _load(csv_file: str, cache_file: str):
    if os.path.exists(cache_file):
        return pd.read_feather(cache_file)
    else:
        return pd.read_csv(csv_file, low_memory=False)


_SPLITS = ["train", "dev", "test"]
_LABELS = ["Label", "attack_cat"]


def load(dir: str = "data/processed/"):
    return {
        k: _load(
            os.path.join(dir, f"{k}.csv"),
            os.path.join(dir, f"{k}.feather"),
        )
        for k in _SPLITS
    }


def must_norm(col):

    if col in _LABELS:
        return False

    if re.match(r".+_[0-9]+$", col):
        return False

    return True


class DataLoader:
    def __init__(self, batch_size=64, device="cuda", W=100):

        self.data = load()
        self.batch_size = batch_size
        self.device = device
        self.W = W

        # Normalize features to zero mean and unit variance

        mean = self.data["train"].mean()
        var = self.data["train"].var()

        for split in self.data:
            for col in self.data[split]:
                if must_norm(col):
                    self.data[split][col] = (self.data[split][col] - mean[col]) / (
                        var[col] + 1e-6
                    ) ** 0.5

        self.cache_x = {}
        self.cache_y = {}

    def len(self, split):
        return self.data[split].shape[0] // self.batch_size // self.W

    def iter(self, split, drop_last=True, W=None):

        if W is None:
            W = self.W

        if split in self.cache_x:
            x = self.cache_x[split]
        else:
            x = self.data[split].drop(columns=_LABELS).to_numpy()

        if split in self.cache_y:
            y = self.cache_y[split]
        else:
            y = self.data[split].loc[:, _LABELS].to_numpy()

        starts = np.random.randint(0, x.shape[0] - W, x.shape[0] // W)

        x = np.stack([x[i : i + W] for i in starts], axis=0)
        y = np.stack([y[i : i + W] for i in starts], axis=0)

        # Shuffle the sequences
        idx = np.random.permutation(x.shape[0])

        x = x[idx]
        y = y[idx]

        # Drop last elements so that the batch size divides evenly
        n = (x.shape[0] // self.batch_size) * self.batch_size

        x, x_rem = x[:n], x[n:]
        y, y_rem = y[:n], y[n:]

        x = x.reshape(-1, self.batch_size, W, x.shape[-1])
        y = y.reshape(-1, self.batch_size, W, y.shape[-1])

        kwargs = {"dtype": torch.float32, "device": self.device}

        for xx, yy in zip(x, y):
            yield torch.tensor(xx, **kwargs), torch.tensor(yy, **kwargs)

        if not drop_last:
            yield torch.tensor(x_rem, **kwargs), torch.tensor(y_rem, **kwargs)


if __name__ == "__main__":

    d = DataLoader()

    for x, y in d.iter("train"):
        print(x.shape, y.shape)
        break
