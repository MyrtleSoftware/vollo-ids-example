import pandas as pd
import os


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


def load(dir: str = ""):

    keys = ["train", "dev", "test"]

    x = {}
    y = {}

    for k in keys:
        x[k], y[k] = _load(os.path.join(dir, f"{k}.csv"), os.path.join(dir, f"{k}.pkl"))

    return x, y


if __name__ == "__main__":
    load()
