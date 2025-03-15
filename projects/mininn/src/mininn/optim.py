from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np

from mininn.parameter import Parameter


class Optimizer(ABC):
    def __init__(self, parameters: Iterable[Parameter]):
        self.parameters = list(parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, parameters: Iterable[Parameter], lr=0.001, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.grad_state: list[Optional[np.ndarray]] = [None] * len(self.parameters)

    def step(self):
        for i, param in enumerate(self.parameters):
            grad = param.grad
            if grad is None:
                continue
            if self.grad_state[i] is not None:
                grad += self.grad_state[i] * self.momentum  # type: ignore
            param.add(-grad, alpha=self.lr)
            if self.momentum > 0:
                self.grad_state[i] = grad
