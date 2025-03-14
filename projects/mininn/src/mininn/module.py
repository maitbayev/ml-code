from abc import ABC, abstractmethod
from typing import Any


class Module(ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def backward(self, *gradients: Any, **kwargs: Any) -> Any:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
