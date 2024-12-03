import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def monadic_int(s: str, base: int):
    try:
        return int(s, base=base)
    except ValueError:
        return None


def str_to_int(s: str):

    if (result := monadic_int(s, 10)) is not None:
        return result

    if (result := monadic_int(s, 8)) is not None:
        return result

    if (result := monadic_int(s, 16)) is not None:
        return result

    return 0


def extractor(t: str):
    if t == "nominal":
        return str
    elif t == "integer" or t == "binary":
        return str_to_int
    elif t == "float":
        return float
    elif t == "timestamp":
        return int
    else:
        raise ValueError(f"Unknown type: {t}")


def short(s: str):

    if len(s) > 50:
        return f"{s[:50]}..."

    return s


# ====== Parse the datasets ======

features = pd.read_csv("data/csv/NUSW-NB15_features.csv", low_memory=False)

names = features["Name"]
types = features["Type "]

# Load the data, using the names as headers

df = pd.concat(
    [
        pd.read_csv(
            f"data/csv/UNSW-NB15_{i}.csv",
            names=names,
            converters={name: extractor(t.lower()) for name, t in zip(names, types)},
        )
        for i in range(1, 5)
    ],
    axis=0,
    ignore_index=True,
)


# Missing attack_cat == benign
df["attack_cat"] = df["attack_cat"].fillna("None")


# Show the nan values
if df.isna().sum().sum() > 0:
    print("Warning: NaN values present")


# Compare the requested and actual data types
for name, act, req, desc in zip(names, df.dtypes, types, features["Description"]):
    print(f"{name:<16} {req:>10} -> {str(act):<10} {short(desc)}")


# ====== Reduce the number of protocols ======

print("\nValue counts of categorical features:")
for name, type in zip(names, types):
    if type == "nominal":
        print(f"{name}: {df[name].nunique()}")


# Get the top k protocols
top_protocols = df["proto"].value_counts().head(20).index


# Replace the rest with "other"
df.loc[~df["proto"].isin(top_protocols), "proto"] = "other"


# ====== One-hot the categorical features ======

# Encode the attack categories
df["attack_cat"] = LabelEncoder().fit_transform(df["attack_cat"])

enc = pd.DataFrame()

for name, type in zip(names, types):
    if type == "nominal" and name != "attack_cat":
        enc[name] = LabelEncoder().fit_transform(df[name])

print(f"\nOne-hot encoding: {[n for n in enc.columns]}")

one_hot = pd.concat(
    [pd.get_dummies(enc[col], prefix=col, dtype=int) for col in enc.columns], axis=1
)

df = pd.concat([one_hot, df.drop(columns=enc.columns)], axis=1)

# ====== Sort by time and drop ======

# Stable sort preserves the order of equal elements
df = df.sort_values(by=["Ltime", "Stime"], kind="mergesort")

df = df.drop(columns=["Ltime", "Stime"])

# ======

print(df)

# Split into 80/10/10 train/dev/test splits

n = len(df) // 10

train, dev, test = (
    df[: 8 * n],
    df[8 * n : 9 * n],
    df[9 * n :],
)

print("Train:", train.shape)
print(train["attack_cat"].value_counts())

print("Dev:", dev.shape)
print(dev["attack_cat"].value_counts())

print("Test:", test.shape)
print(test["attack_cat"].value_counts())

# Save the data
os.makedirs("data/processed", exist_ok=True)

train.to_csv("data/processed/train.csv", index=False)
dev.to_csv("data/processed/dev.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

# Save as binary

train.to_feather("data/processed/train.feather")
dev.to_feather("data/processed/dev.feather")
test.to_feather("data/processed/test.feather")
