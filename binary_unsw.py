import torch
from load_unsw import DataLoader
from tqdm import tqdm
from model import Net

# Set the pytorch seed for reproducibility
# torch.manual_seed(3)


@torch.no_grad()
def eval(model, iter, eps=1e-6):

    model.eval()

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for x, y in iter:

        pred, _ = model(x)

        pred = pred > 0.5
        target = y[:, :, :1] > 0.5

        tp += (pred & target).sum().item()
        fp += (pred & ~target).sum().item()
        tn += (~pred & ~target).sum().item()
        fn += (~pred & target).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    t = max(min(tp, tn, fp, fn), 1)

    model.train()

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

model = Net(input_size=180).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters())//1000}k")
print(model)

# =================

optimizer = torch.optim.AdamW(model.parameters())

# =================

loader = DataLoader(device=device)


for i in range(10):
    for x, y in (
        t := tqdm(loader.iter("train"), leave=False, total=loader.len("train"))
    ):

        optimizer.zero_grad()

        logits, _ = model(x)
        # The first column of y is attack/!attack
        y = y[:, :, :1]

        loss = torch.nn.functional.binary_cross_entropy(logits, y)

        # print(loss.item())

        loss.backward()
        optimizer.step()

        # for p in optimizer.param_groups:
        #     p["lr"] = p["lr"] * 0.99

        t.set_description(f"Loss: {loss.item():.4f}")

    print(f"Dev set - epoch {i}:")

    for k, v in eval(model, loader.iter("dev", drop_last=False)).items():
        print(f"\t{k}: {v}")

print("Test set:")


for w in [1, 10, 50, 100, 500, 1000, 10000]:
    for k, v in eval(model, loader.iter("test", drop_last=False, W=w)).items():
        print(f"\t{w:>4}: {k}: {v}")


# Save the model
torch.save(model.state_dict(), "build/model.pt")
