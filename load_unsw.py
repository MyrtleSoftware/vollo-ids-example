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
    def __init__(self, batch_size=1024, device="cuda"):

        self.data = load()
        self.batch_size = batch_size
        self.device = device

        # Normalize features to zero mean and unit variance

        mean = self.data["train"].mean()
        var = self.data["train"].var()

        for split in self.data:
            for col in self.data[split]:
                if must_norm(col):
                    self.data[split][col] = (self.data[split][col] - mean[col]) / (
                        var[col] + 1e-6
                    ) ** 0.5

    def len(self, split):
        return self.data[split].shape[0] // self.batch_size

    def iter(self, split, drop_last=True):

        W = 50

        x = self.data[split].drop(columns=_LABELS).to_numpy()
        y = self.data[split].loc[:, _LABELS].to_numpy()

        if len(x) > (n := self.batch_size * 300):
            x = x[:n]
            y = y[:n]

        # x = x[:10000]
        # y = y[:10000]

        # print(x.shape)

        x = np.stack([x[i : i - W] for i in range(W)], axis=1)
        y = np.stack([y[i : i - W] for i in range(W)], axis=1)

        # print(x.shape)

        # Shuffle in the first dimension

        idx = np.random.permutation(x.shape[0])

        x = x[idx]
        y = y[idx]

        # print(x.shape)

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
