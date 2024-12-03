import torch
import vollo_torch

from torch import nn
from vollo_torch import nn as vv


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

    def compile(self):

        # Cache mode
        mode = self.training
        self.eval()

        dummy = torch.randn(1, 1, self.input_size)

        # Trace
        model, _ = vollo_torch.fx.prepare_shape(self, dummy)

        nnir, _ = vollo_torch.fx.nnir.to_nnir(model).streaming_transform(1)

        # Restore mode
        self.train(mode)

        return nnir, dummy[:, 0, :]


if __name__ == "__main__":

    model = Net(180)

    print("Parameters:", sum(p.numel() for p in model.parameters()) // 1000, "k")

    print(model)

    nnir, dummy_input = model.compile()

    import vollo_compiler

    config = vollo_compiler.Config.ia_420f_c6b32()

    program = nnir.to_program(config)

    print(program.metrics())

    vm = program.to_vm()
    vm_output = vm.run(dummy_input.detach().numpy())

    print("cycle count:", vm.cycle_count())
    print(f"latency (compute): {vm.compute_duration_us():.1f}us")
