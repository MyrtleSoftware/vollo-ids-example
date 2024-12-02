import kagglehub
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder


print("Downloading dataset...")

path = kagglehub.dataset_download(
    "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot"
)

print("Loading dataset...")

path = os.path.join(
    path, "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
)

df = pd.read_csv(path, low_memory=False)

print("Columns:")
for col in df.columns:
    print("\t", col)

print("Encoding categorical features...")

keys = [
    "http.request.method",
    "http.referer",
    "http.request.version",
    "dns.qry.name.len",
    "mqtt.conack.flags",
    "mqtt.protoname",
    "mqtt.topic",
]

enc = pd.DataFrame()

for key in keys:
    enc[key] = LabelEncoder().fit_transform(df[key])

print("One-hot encoding...")

one_hot = pd.concat(
    [pd.get_dummies(enc[key], prefix=key, dtype=float) for key in keys], axis=1
)

df = df.drop(columns=keys)
df = pd.concat([df, one_hot], axis=1)

print("Cleaning data...")

# This sort is needed for later time-based splits
df = df.sort_values("frame.time")

print("Dropping columns...")

drop_columns = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "icmp.transmit_timestamp",
    "http.request.uri.query",
    "tcp.options",
    "tcp.payload",
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "mqtt.msg",
]

df.drop(drop_columns, axis=1, inplace=True)
df.dropna(axis=0, how="any", inplace=True)
df.drop_duplicates(subset=None, keep="first", inplace=True)

print(df["Attack_type"].value_counts())

print("Feature size:", len(df.columns) - 2)

print("Making splits...")

# Shuffle the rows

# df = df.sample(frac=1).reset_index(drop=True)

# 70/10/20 splits

x, y = 7, 8

n = len(df) // 10

train = df[: x * n]
dev = df[x * n : y * n]
test = df[y * n :]

print("\nTrain")
print(train["Attack_type"].value_counts())
print("\nDev")
print(dev["Attack_type"].value_counts())
print("\nTest")
print(test["Attack_type"].value_counts())

print("Saving splits...")

if not os.path.exists("build"):
    os.makedirs("build")

train.to_csv("build/train.csv", index=False)
dev.to_csv("build/dev.csv", index=False)
test.to_csv("build/test.csv", index=False)

train.to_pickle("build/train.pkl")
dev.to_pickle("build/dev.pkl")
test.to_pickle("build/test.pkl")
