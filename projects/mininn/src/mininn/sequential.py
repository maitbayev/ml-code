import numpy as np

from mininn.module import Module


class Sequential(Module):
    def __init__(self, modules: list[Module]):
        super().__init__()
        self.modules = modules

    def forward(self, input: np.ndarray) -> np.ndarray:
        output = input
        for module in self.modules:
            output = module(output)
        return output

    def backward(self, gradients: np.ndarray) -> np.ndarray:
        output = gradients
        for module in reversed(self.modules):
            output = module.backward(output)
        return output
