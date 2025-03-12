from abc import ABC, abstractmethod
from typing import Any


class Function(ABC):
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def backward(self, *gradients: Any) -> Any:
        pass
