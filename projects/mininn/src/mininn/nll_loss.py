import numpy as np
from mininn import Function


class NLLLoss(Function):
    def __init__(self):
        super().__init__()
        self.target = None
        self.input_shape = None

    def forward(self, input, target):
        batches = input.shape[0]
        # input (B, N)
        # target (N,)
        sum = 0
        for b in range(batches):
            sum += -input[b][target[b]]
        self.target = target
        self.input_shape = input.shape
        return sum / batches

    def backward(self) -> np.ndarray:
        batches = self.input_shape[0]
        output = np.zeros(self.input_shape)
        for b in range(batches):
            output[b][self.target[b]] = -1 / batches
        return output
