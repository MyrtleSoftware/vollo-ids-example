from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np
import time


from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

print("Loading dataset...")

keys = ["train", "val", "test"]

splits = {k: pd.read_csv(f"{k}.csv", low_memory=False) for k in keys}

print("Pre-process dataset...")

attacks = {
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

X = {}
Y = {}

for k in keys:
    splits[k]["Attack_type"] = splits[k]["Attack_type"].map(attacks)
    X[k] = splits[k].drop(columns=["Attack_label", "Attack_type"])
    Y[k] = splits[k]["Attack_label"]


# scaler = StandardScaler().fit(X["train"])

# for k in keys:
#     X[k] = scaler.transform(X[k])

for k, v in X.items():
    print(k, v.shape)

print("Building model...")


def cnn_lstm_gru_model(input_shape, num_classes):

    model = Sequential(
        [
            Conv1D(
                filters=32, kernel_size=3, activation="relu", input_shape=input_shape
            ),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            LSTM(64, return_sequences=True),
            LSTM(64, return_sequences=False),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="sigmoid"),
        ]
    )

    # MLP
    model = Sequential(
        [
            Input(shape=input_shape),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


input_shape = (X["train"].shape[1], 1)
num_classes = 1
model = cnn_lstm_gru_model(input_shape, num_classes)
model.summary()
# plot_model(model)

## Train model


train_start_time = time.time()
# Train the model
history = model.fit(
    X["train"],
    Y["train"],
    validation_data=(X["val"], Y["val"]),
    epochs=2,
    batch_size=1024,
)
# Record the ending time
train_end_time = time.time()

# Record the starting time for testing
test_start_time = time.time()
# Evaluate the model
loss, accuracy = model.evaluate(X["test"], Y["test"], batch_size=128)
# Record the ending time for testing
test_end_time = time.time()

print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Calculate and print the training time
train_time = train_end_time - train_start_time
print(f"Training time: {train_time:.2f} seconds")

# Calculate and print the testing time
test_time = test_end_time - test_start_time
print(f"Testing time: {test_time:.2f} seconds")
