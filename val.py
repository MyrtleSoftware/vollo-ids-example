from enum import Enum
import itertools

import numpy as np
import torch

import vollo_compiler
import vollo_torch

from model import Net
from loader import DataLoader


class RunOnVollo(Enum):
    NO_VOLLO = 0
    VOLLO_VM = 1
    VOLLO_ACCELERATOR = 2


class Metrics:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def add_sample(self, pred, target):
        self.tp += (pred & target).sum().item()
        self.fp += (pred & ~target).sum().item()
        self.tn += (~pred & ~target).sum().item()
        self.fn += (~pred & target).sum().item()

    def summary(self, eps):
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        t = self.tp + self.fp + self.tn + self.fn + eps

        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Confusion-matrix": [
                [self.tp / t, self.fp / t],
                [self.fn / t, self.tn / t],
            ],
        }


@torch.no_grad()
def eval(model, iter, orig_device, run_on_vollo=RunOnVollo.NO_VOLLO, eps=1e-6):
    model.eval()

    metrics = Metrics()

    if run_on_vollo in [RunOnVollo.VOLLO_VM, RunOnVollo.VOLLO_ACCELERATOR]:
        model = model.cpu()
        _, prog, _ = model.compile()

        # Will only be used if using the Vollo VM
        vm = prog.to_vm()

        iter = (
            itertools.islice(iter, 1) if run_on_vollo == RunOnVollo.VOLLO_VM else iter
        )

        for x, y in iter:
            x = x.cpu()
            y = y.cpu()

            pred = []

            for batch_ix in range(x.size(0)):
                if run_on_vollo == RunOnVollo.VOLLO_VM:
                    stream = x[batch_ix : batch_ix + 1].numpy()
                    pred.append(torch.from_numpy(vm.run_timesteps(stream, 1, 1)))

                # Run on accelerator
                else:
                    # Conditionally import vollo_rt so as to not require vollo_rt.so if we're not using it
                    import vollo_rt

                    # Create a new Vollo context for each stream in the batch so that state
                    # is not being reused
                    with vollo_rt.VolloRTContext() as ctx:
                        ctx.add_accelerator(0)
                        ctx.load_program(prog)

                        seq_pred = []
                        for timestep_ix in range(x.size(1)):
                            elem = x[batch_ix : batch_ix + 1, timestep_ix, :]
                            seq_pred.append(ctx.run(elem.detach()))
                        seq_pred = torch.stack(seq_pred, axis=1)

                    pred.append(seq_pred)

            pred = torch.cat(
                pred,
                axis=0,
            )

            pred = pred > 0.5
            target = y[:, :, :1] > 0.5

            metrics.add_sample(pred, target)

        model = model.to(orig_device)

    else:
        for x, y in iter:
            pred = model(x)

            pred = pred > 0.5
            target = y[:, :, :1] > 0.5

            metrics.add_sample(pred, target)

    model.train()

    return metrics.summary(eps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    run_on_vollo_group = parser.add_mutually_exclusive_group()
    run_on_vollo_group.add_argument("--vollo-vm", action="store_true")
    run_on_vollo_group.add_argument("--vollo", action="store_true")
    args = parser.parse_args()

    # Default
    run_on_vollo = RunOnVollo.NO_VOLLO
    if args.vollo_vm:
        run_on_vollo = RunOnVollo.VOLLO_VM
    elif args.vollo:
        run_on_vollo = RunOnVollo.VOLLO_ACCELERATOR

    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(device=device)

    model = Net(input_size=180)
    model.load_state_dict(torch.load("build/model.pt", weights_only=True, map_location=device))
    model = model.to(device)

    print("Test set:")
    for k, v in eval(
        model, loader.iter("test", drop_last=False), device, run_on_vollo=run_on_vollo
    ).items():
        print(f"\t{k}: {v}")
