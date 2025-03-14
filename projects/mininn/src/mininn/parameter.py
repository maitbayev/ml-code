from typing import Optional

import numpy as np


class Parameter:
    def __init__(self, value: np.ndarray):
        self.value = value
        self.grad: Optional[np.ndarray] = None

    @classmethod
    def uniform(cls, shape: tuple | int, range: float) -> "Parameter":
        return Parameter(np.random.uniform(low=-range, high=range, size=shape))

    def accumulate_grad(self, grad: np.ndarray):
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
