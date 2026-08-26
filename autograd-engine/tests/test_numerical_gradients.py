"""Second, independent check: analytic gradients against central differences.

PyTorch and this engine could in principle share a conceptual mistake. Central
finite differences depend on nothing but the forward pass, so agreement here
confirms the backward pass really is the derivative of the forward pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from tinygrad_core import Tensor


def numerical_gradient(function, array: np.ndarray, step: float = 1e-6) -> np.ndarray:
    gradient = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        shifted_up = array.copy()
        shifted_down = array.copy()
        shifted_up[index] += step
        shifted_down[index] -= step
        gradient[index] = (function(shifted_up) - function(shifted_down)) / (2 * step)
    return gradient


def check(build, array: np.ndarray, tolerance: float = 1e-6) -> None:
    tensor = Tensor(array, requires_grad=True)
    build(tensor).sum().backward()
    expected = numerical_gradient(lambda a: build(Tensor(a)).sum().data, array)
    np.testing.assert_allclose(tensor.grad, expected, atol=tolerance)


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_polynomial(rng):
    check(lambda x: x**3.0 + x * 2.0, rng.normal(size=(4, 3)))


def test_tanh_composition(rng):
    check(lambda x: (x.tanh() * x).sum(axis=1), rng.normal(size=(5, 4)))


def test_log_softmax(rng):
    check(lambda x: x.log_softmax(axis=-1) * 3.0, rng.normal(size=(4, 5)))


def test_exp_log_chain(rng):
    check(lambda x: (x.exp() + 1.0).log(), rng.normal(size=(4, 4)))


def test_mean_over_axis(rng):
    check(lambda x: x.mean(axis=0) ** 2.0, rng.normal(size=(6, 3)))


def test_diamond_shaped_graph_accumulates_both_paths(rng):
    """One tensor feeding two branches that later merge: the reverse pass must
    sum the contributions from both paths rather than overwrite one."""
    check(lambda x: x.tanh() * x.relu(), rng.normal(size=(5, 4)))
