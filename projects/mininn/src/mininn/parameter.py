from typing import Optional

import numpy as np


class Parameter:
    def __init__(self, value: np.ndarray):
        self.value = value
        self.grad: Optional[np.ndarray] = None

    @classmethod
    def uniform(cls, shape: tuple | int, range: float) -> "Parameter":
        return Parameter(np.random.uniform(low=-range, high=range, size=shape))

    def set(self, value: np.ndarray) -> "Parameter":
        self.value = value
        return self

    def add(self, delta: np.ndarray, alpha: float = 1) -> "Parameter":
        self.value += alpha * delta
        return self

    def accumulate_grad(self, delta: np.ndarray):
        if self.grad is None:
            self.grad = delta.copy()
        else:
            self.grad += delta
