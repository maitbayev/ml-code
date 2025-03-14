from abc import ABC
from mininn.function import Function


class Module(Function, ABC):
    def __init__(self):
        super().__init__()
