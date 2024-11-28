import kagglehub
import os
import pandas as pd

from hashlib import md5
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

# Print column names
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

df = df.drop_duplicates()


# Function to create a hash for each column
def hash_column(series):
    return md5(pd.util.hash_pandas_object(series, index=False).values).hexdigest()


# Function to find columns with identical hashes
def find_identical_columns_by_hash(df):

    hash_dict = {}

    for col in df.columns:

        col_hash = hash_column(df[col])

        if col_hash in hash_dict:
            hash_dict[col_hash].append(col)
        else:
            hash_dict[col_hash] = [col]

    return [cols for cols in hash_dict.values() if len(cols) > 1]


# Applying the function to the DataFrame
identical_column_groups = find_identical_columns_by_hash(df)

# Sort by time for dev/test splits later
df = df.sort_values("frame.time")


print("Dropping columns...")

print("Groups of identical columns:")
for group in identical_column_groups:
    print("\t", group)
    df = df.drop(group[1:], axis=1)


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

print("Making splits...")

# 70/10/20 splits in the time domain

n = len(df) // 10

train = df[: 7 * n]
dev = df[7 * n : 8 * n]
test = df[8 * n :]

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
