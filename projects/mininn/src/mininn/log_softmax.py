from numpy.typing import NDArray
from mininn.function import Function

import numpy as np


class LogSoftmax(Function):
    def __init__(self):
        self.e = np.array([])
        self.log_softmax = np.array([])

    def forward(self, input: NDArray) -> NDArray:
        # log(e[i]/sum e[j])
        #   = log(e^x[i]) - log(sum e^x[j])
        #   = x[i] - log(sum e^x[j])
        #   = x[i] - log(e^mx * sum  e^(x[j]-mx))
        #   = x[i] - mx - log(sum e^(x[j]-max))

        # (B, 1)
        mx = np.max(input, axis=1, keepdims=True)
        # (B, N)
        self.e = np.exp(input - mx)
        self.log_softmax = input - mx - np.log(np.sum(self.e, axis=1, keepdims=True))
        return self.log_softmax

    def backward(self, gradients: NDArray) -> NDArray:
        # if i != j:
        #   d log(e[i]/sum )  / d x[j]
        #       = - sum / e[i] * (e[i] / sum) * (e[j] / sum)
        #       = - e[j] / sum
        # if i == j:
        #   d log(e[i]/sum ) / d x[j]
        #       = sum / e[i] * (e[i]/sum) * (1 - e[i]/sum)
        #       = 1 - e[i] / sum

        batches = self.log_softmax.shape[0]
        results = []

        for b in range(batches):
            e = self.e[b : b + 1]
            j = np.eye(e.shape[1]) - e / e.sum()
            results.append(gradients[b] @ j)
        return np.array(results)
