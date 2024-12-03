import torch
import vollo_torch
import vollo_compiler

from torch import nn
from vollo_torch import nn as vv


def eval(fn):

    @torch.no_grad()
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out

    return inner


class Net(nn.Module):
    def __init__(
        self,
        input_size,
        hid: int = 64,
        dropout: float = 0.1,
    ):

        self.input_size = input_size

        super().__init__()

        kwargs = {
            "hidden_size": hid,
            "batch_first": True,
        }

        self.post = nn.Sequential(
            vv.LSTM(input_size=input_size, **kwargs),
            nn.Dropout(p=dropout),
            vv.LSTM(input_size=hid, **kwargs),
            nn.Dropout(p=dropout),
            nn.Linear(hid, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, H]

        Returns:
            r: [B, T, 1] on interval [0, 1]
        """
        return self.post(x).clamp(0, 1)

    @eval
    def compile(self, config=vollo_compiler.Config.ia_420f_c6b32()):

        dummy = torch.randn(1, 1, self.input_size)

        model, _ = vollo_torch.fx.prepare_shape(self, dummy)

        nnir, _ = vollo_torch.fx.nnir.to_nnir(model).streaming_transform(1)

        program = nnir.to_program(config)

        vm = program.to_vm()

        # The streaming transform removes the time-dimension
        vm.run(dummy[:, 0, :].detach().numpy())

        stats = {
            "cycle_count": vm.cycle_count(),
            "compute_latency/us": vm.compute_duration_us(),
        }

        return nnir, program, stats


if __name__ == "__main__":

    model = Net(180)

    print("Parameters:", sum(p.numel() for p in model.parameters()) // 1000, "k")

    print(model)

    _, prog, stats = model.compile()

    print(stats)

    print(prog.metrics())
