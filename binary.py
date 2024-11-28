import torch
from load import DataLoader
from tqdm import tqdm
from model import Net


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


for _ in range(1):
    for x, y in tqdm(loader.iter("train"), leave=False):

        optimizer.zero_grad()

        logits = model(x[:, :, None])
        # The first column of y is attack/!attack
        loss = torch.nn.functional.binary_cross_entropy(logits, y[:, :1])

        loss.backward()
        optimizer.step()

    print("dev set", eval(loader.iter("dev", drop_last=False)))

print("test set", eval(loader.iter("test", drop_last=False)))

# Save the model
torch.save(model.state_dict(), "build/model.pt")
