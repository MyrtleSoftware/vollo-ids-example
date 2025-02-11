import copy
import os

import torch
from tqdm import tqdm

from model import Net
from loader import DataLoader
from val import eval

"""
Time series analysis on the UNSW-NB15 dataset:
    Deep Learning for Intrusion Detection Systems (IDSs) in Time Series Data
"""


@torch.no_grad()
def update_ema(model, ema_model, alpha=0.999):
    for p, ema_p in zip(model.parameters(), ema_model.parameters()):
        ema_p.set_(alpha * ema_p.data + (1 - alpha) * p.data)


# Set the pytorch seed for reproducibility
torch.manual_seed(42)

# =================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =================

model = Net(input_size=180)
ema_model = copy.deepcopy(model)

print(f"Parameters: {sum(p.numel() for p in model.parameters())//1000}k")

_, prog, vollo_stats = model.compile()

print(f"{vollo_stats=}")

print(model)

model = model.to(device)
ema_model = ema_model.to(device)

# =================

optimizer = torch.optim.AdamW(model.parameters())

# =================

loader = DataLoader(device=device)

for i in range(10):
    for x, y in (
        t := tqdm(loader.iter("train"), leave=False, total=loader.len("train"))
    ):
        optimizer.zero_grad()

        probs = model(x)
        # The first column of y is attack/!attack
        y = y[:, :, :1]

        loss = torch.nn.functional.binary_cross_entropy(probs, y)

        loss.backward()
        optimizer.step()

        t.set_description(f"Loss: {loss.item():.4f}")

        update_ema(model, ema_model)

    for p in optimizer.param_groups:
        p["lr"] = max(1e-5, p["lr"] * 0.9)

    print(f"Dev set - epoch {i}:")
    for k, v in eval(ema_model, loader.iter("dev", drop_last=False), device).items():
        print(f"\t{k}: {v}")

print("Test set:")
for k, v in eval(ema_model, loader.iter("test", drop_last=False), device).items():
    print(f"\t{k}: {v}")


# Save the model

os.makedirs("build", exist_ok=True)

torch.save(model.state_dict(), "build/model.pt")
