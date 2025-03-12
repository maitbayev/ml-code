import mininn
from mininn.log_softmax import LogSoftmax
from mininn.nll_loss import NLLLoss
import numpy as np


class CrossEntropyLoss(mininn.Function):
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.nll_loss = NLLLoss()

    def forward(self, input, target):
        return self.nll_loss.forward(self.log_softmax(input), target)

    def backward(self) -> np.ndarray:
        g = self.nll_loss.backward()
        return self.log_softmax.backward(g)
