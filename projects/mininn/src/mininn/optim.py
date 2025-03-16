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
        self.grad_avg: list[Optional[np.ndarray]] = [None] * len(self.parameters)

    def step(self):
        for i, param in enumerate(self.parameters):
            grad = param.grad
            if grad is None:
                continue
            if self.grad_avg[i] is not None:
                grad += self.grad_avg[i] * self.momentum  # type: ignore
            param.add(-grad, alpha=self.lr)
            if self.momentum > 0:
                self.grad_avg[i] = grad


class RMSprop(Optimizer):
    def __init__(
        self, parameters: Iterable[Parameter], lr=0.001, alpha=0.99, eps=1e-08
    ):
        super().__init__(parameters)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.square_avg: list[Optional[np.ndarray]] = [None] * len(self.parameters)

    def step(self):
        for i, param in enumerate(self.parameters):
            grad = param.grad
            if grad is None:
                continue
            sqr_avg = self.square_avg[i]
            if sqr_avg is None:
                sqr_avg = np.zeros_like(grad)
            sqr_avg = self.alpha * sqr_avg + (1 - self.alpha) * grad * grad
            param.add(-grad / (np.sqrt(sqr_avg) + self.eps), alpha=self.lr)
            self.square_avg[i] = sqr_avg
