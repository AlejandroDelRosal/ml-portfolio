from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Module:
    def parameters(self) -> list[Tensor]:
        collected: list[Tensor] = []
        for value in vars(self).values():
            if isinstance(value, Tensor) and value.requires_grad:
                collected.append(value)
            elif isinstance(value, Module):
                collected.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        collected.extend(item.parameters())
        return collected

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # Kaiming initialization (He et al. 2015, arXiv:1502.01852): scaling the
        # weight variance by 2/fan_in keeps activation variance stable through
        # ReLU layers, which halve the variance of their input.
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(rng.normal(0, scale, (in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias


class MLP(Module):
    def __init__(self, sizes: list[int], seed: int | None = None):
        self.layers = [
            Linear(sizes[i], sizes[i + 1], seed=None if seed is None else seed + i)
            for i in range(len(sizes) - 1)
        ]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)

    def parameters(self) -> list[Tensor]:
        return [p for layer in self.layers for p in (layer.weight, layer.bias)]


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean negative log-likelihood of the correct class."""
    log_probabilities = logits.log_softmax(axis=-1)
    rows = np.arange(len(targets))
    return -log_probabilities[rows, targets].mean()


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    return ((predictions - targets) ** 2).mean()
