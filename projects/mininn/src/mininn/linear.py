from mininn.module import Module
from mininn.parameter import Parameter
import numpy as np


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        lim = 1 / np.sqrt(in_features)
        self.weight = Parameter.uniform((in_features, out_features), lim)
        self.bias = Parameter.uniform(out_features, lim) if bias else None

    def forward(self, input: np.ndarray) -> np.ndarray:
        # input (B, N) x (N, M) + (1, B)
        out = input @ self.weight.value
        if self.bias is not None:
            out += self.bias.value.reshape((1, -1))
        self.input = input
        return out

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        self.weight.grad = np.dot(self.input.T, gradients)
        if self.bias:
            self.bias.grad = gradients.sum(axis=0)
        return np.ndarray([])
