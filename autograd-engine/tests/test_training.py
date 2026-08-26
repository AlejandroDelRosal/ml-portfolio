from __future__ import annotations

import numpy as np
import pytest

from tinygrad_core import Tensor, MLP, Linear, SGD, Adam, mse_loss


def test_sgd_reaches_the_analytic_least_squares_solution():
    """Linear regression has a closed-form optimum, so a correct optimizer on a
    correct gradient must converge to it."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    true_weights = np.array([[2.0], [-1.0], [0.5]])
    y = x @ true_weights

    layer = Linear(3, 1, seed=0)
    optimizer = SGD(layer.parameters(), learning_rate=0.05)
    for _ in range(2000):
        optimizer.zero_grad()
        mse_loss(layer(Tensor(x)), Tensor(y)).backward()
        optimizer.step()

    np.testing.assert_allclose(layer.weight.data, true_weights, atol=1e-3)
    np.testing.assert_allclose(layer.bias.data, [0.0], atol=1e-3)


def test_network_learns_xor_a_problem_no_linear_model_can_solve():
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])

    model = MLP([2, 8, 1], seed=3)
    optimizer = Adam(model.parameters(), learning_rate=0.05)
    for _ in range(2000):
        optimizer.zero_grad()
        mse_loss(model(Tensor(x)), Tensor(y)).backward()
        optimizer.step()

    predictions = model(Tensor(x)).data
    assert np.all((predictions > 0.5) == (y > 0.5))


def test_adam_bias_correction_makes_the_first_step_full_sized():
    """Without bias correction the first Adam step would be scaled down by
    roughly (1 - beta1), since both moment estimates start at zero."""
    parameter = Tensor(np.array([1.0]), requires_grad=True)
    optimizer = Adam([parameter], learning_rate=0.1)
    parameter.grad = np.array([1.0])
    optimizer.step()
    assert parameter.data[0] == pytest.approx(0.9, abs=1e-6)


def test_zero_grad_prevents_accumulation_across_steps():
    parameter = Tensor(np.array([2.0]), requires_grad=True)
    (parameter * 3.0).sum().backward()
    first = parameter.grad.copy()

    (parameter * 3.0).sum().backward()
    assert parameter.grad == pytest.approx(first * 2)

    parameter.zero_grad()
    (parameter * 3.0).sum().backward()
    assert parameter.grad == pytest.approx(first)


def test_gradients_do_not_flow_into_tensors_that_do_not_require_them():
    trainable = Tensor(np.ones((2, 2)), requires_grad=True)
    constant = Tensor(np.ones((2, 2)), requires_grad=False)
    (trainable * constant).sum().backward()
    assert trainable.grad is not None
    assert constant.grad is None


def test_backward_on_a_non_scalar_output_is_rejected():
    tensor = Tensor(np.ones((2, 2)), requires_grad=True)
    with pytest.raises(RuntimeError):
        (tensor * 2.0).backward()
