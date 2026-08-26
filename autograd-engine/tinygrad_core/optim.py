from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Optimizer:
    def __init__(self, parameters: list[Tensor]):
        self.parameters = parameters

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, parameters: list[Tensor], learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        for parameter, velocity in zip(self.parameters, self.velocities):
            if parameter.grad is None:
                continue
            velocity *= self.momentum
            velocity += parameter.grad
            parameter.data -= self.learning_rate * velocity


class Adam(Optimizer):
    """Kingma & Ba 2015, arXiv:1412.6980."""

    def __init__(
        self,
        parameters: list[Tensor],
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.first_moments = [np.zeros_like(p.data) for p in parameters]
        self.second_moments = [np.zeros_like(p.data) for p in parameters]
        self.timestep = 0

    def step(self) -> None:
        self.timestep += 1
        for parameter, first, second in zip(self.parameters, self.first_moments, self.second_moments):
            if parameter.grad is None:
                continue
            first *= self.beta1
            first += (1 - self.beta1) * parameter.grad
            second *= self.beta2
            second += (1 - self.beta2) * parameter.grad**2

            # Both moment estimates start at zero and are therefore biased
            # toward zero early in training; these corrections remove that bias.
            first_corrected = first / (1 - self.beta1**self.timestep)
            second_corrected = second / (1 - self.beta2**self.timestep)

            parameter.data -= self.learning_rate * first_corrected / (np.sqrt(second_corrected) + self.eps)
