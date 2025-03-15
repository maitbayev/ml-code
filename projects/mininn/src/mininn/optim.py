from abc import ABC, abstractmethod
from typing import Iterable

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
    def __init__(self, parameters: Iterable[Parameter], lr=0.001):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue
            p.add(-p.grad, alpha=self.lr)
