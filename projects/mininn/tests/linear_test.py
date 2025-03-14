from typing import Optional

import numpy as np
import torch
from pytest import approx

from mininn import Linear


def _torch_linear(
    input: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray]
) -> np.ndarray:
    return torch.nn.functional.linear(
        torch.tensor(input),
        weight=torch.tensor(weight.T),
        bias=None if bias is None else torch.tensor(bias),
    ).numpy(force=True)


def _torch_linear_gradw(
    input: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray], grad: np.ndarray
) -> np.ndarray:
    input = torch.tensor(input)
    weight = torch.tensor(weight.T, requires_grad=True)
    grad = torch.tensor(grad)
    bias = torch.tensor(bias)
    output = torch.nn.functional.linear(input, weight, bias)
    output.backward(grad)
    return weight.grad.numpy(force=True).T


def _torch_linear_gradb(
    input: np.ndarray, weight: np.ndarray, bias: np.ndarray, grad: np.ndarray
) -> np.ndarray:
    input = torch.tensor(input)
    weight = torch.tensor(weight.T)
    grad = torch.tensor(grad)
    bias = torch.tensor(bias, requires_grad=True)
    output = torch.nn.functional.linear(input, weight, bias)
    output.backward(grad)
    return bias.grad.numpy(force=True)


def _torch_linear_grad_input(
    input: np.ndarray, weight: np.ndarray, bias: np.ndarray, grad: np.ndarray
) -> np.ndarray:
    input = torch.tensor(input, requires_grad=True)
    weight = torch.tensor(weight.T)
    grad = torch.tensor(grad)
    bias = torch.tensor(bias)
    output = torch.nn.functional.linear(input, weight, bias)
    output.backward(grad)
    return input.grad.numpy(force=True)


def test_forward():
    for i in range(100):
        in_features, out_features = 3, 5
        if i % 2 == 0:
            in_features, out_features = out_features, in_features
        linear = Linear(in_features, out_features)
        input = np.random.randn(10, in_features)
        assert linear.forward(input) == approx(
            _torch_linear(input, linear.weight.value, linear.bias.value)  # type: ignore
        )


def test_backward_weight():
    for i in range(10):
        batches = 10
        in_features, out_features = 3, 5
        if i % 2 == 0:
            in_features, out_features = out_features, in_features
        linear = Linear(in_features, out_features)
        input = np.random.randn(batches, in_features)
        grad = np.random.randn(batches, out_features)
        linear.forward(input)
        linear.backward(grad)
        assert linear.weight.grad == approx(
            _torch_linear_gradw(input, linear.weight.value, linear.bias.value, grad)  # type: ignore
        )


def test_backward_bias():
    for i in range(10):
        batches = 10
        in_features, out_features = 3, 5
        if i % 2 == 0:
            in_features, out_features = out_features, in_features
        linear = Linear(in_features, out_features)
        input = np.random.randn(batches, in_features)
        grad = np.random.randn(batches, out_features)
        linear.forward(input)
        linear.backward(grad)
        assert linear.bias.grad == approx(
            _torch_linear_gradb(input, linear.weight.value, linear.bias.value, grad)  # type: ignore
        )


def test_backward_input():
    for i in range(10):
        batches = 10
        in_features, out_features = 3, 5
        if i % 2 == 0:
            in_features, out_features = out_features, in_features
        linear = Linear(in_features, out_features)
        input = np.random.randn(batches, in_features)
        grad = np.random.randn(batches, out_features)
        linear.forward(input)
        assert linear.backward(grad) == approx(
            _torch_linear_grad_input(
                input, linear.weight.value, linear.bias.value, grad
            )  # type: ignore
        )
