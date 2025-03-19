import numpy as np
import torch
from pytest import approx

import mininn


def _torch_conv(
    conv: mininn.Conv2d,
    torch_conv: torch.nn.Conv2d,
    input: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        torch_conv.weight.copy_(torch.tensor(conv.weight.value, dtype=torch.float32))
        if conv.bias and torch_conv.bias is not None:
            torch_conv.bias.copy_(torch.tensor(conv.bias.value, dtype=torch.float32))
    input_tensor = torch.tensor(input, dtype=torch.float32)
    return torch_conv(input_tensor).numpy(force=True)


def _check_forward(conv, torch_conv):
    for _ in range(10):
        inp = np.random.randn(1, conv.in_channels, 6, 8)
        assert conv(inp) == approx(_torch_conv(conv, torch_conv, inp), abs=1e-6)


def test_forward():
    conv = mininn.Conv2d(4, 5, 3)
    torch_conv = torch.nn.Conv2d(4, 5, 3)
    _check_forward(conv, torch_conv)


def test_forward_padding_same():
    conv = mininn.Conv2d(4, 5, 3, padding="same")
    torch_conv = torch.nn.Conv2d(4, 5, 3, padding="same")
    _check_forward(conv, torch_conv)
