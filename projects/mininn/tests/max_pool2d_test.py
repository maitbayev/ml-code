import random

import numpy as np
import torch
from pytest import approx

import mininn


def test_forward():
    for _ in range(10):
        k = random.randint(1, 3) * 2 + 1
        stride = random.randint(2, 5)
        padding = random.randint(0, k // 2)
        inp = np.random.rand(4, 5, np.random.randint(10, 20), np.random.randint(10, 20))
        max_pool2d = mininn.MaxPool2D(k, stride, padding)
        inp_torch = torch.tensor(inp)
        max_pool2d_torch = torch.nn.functional.max_pool2d(
            inp_torch, k, stride, padding=padding
        )
        assert max_pool2d(inp) == approx(max_pool2d_torch.numpy(force=True))
