"""Every gradient in the engine, checked against PyTorch's autograd."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tinygrad_core import Tensor


def assert_matches_torch(build_ours, build_torch, inputs, tolerance=1e-9):
    ours = [Tensor(array, requires_grad=True) for array in inputs]
    build_ours(*ours).sum().backward()

    theirs = [torch.tensor(array, dtype=torch.float64, requires_grad=True) for array in inputs]
    build_torch(*theirs).sum().backward()

    for our_tensor, their_tensor in zip(ours, theirs):
        np.testing.assert_allclose(our_tensor.grad, their_tensor.grad.numpy(), atol=tolerance)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_add_and_multiply(rng):
    a, b = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    assert_matches_torch(lambda x, y: x * y + x, lambda x, y: x * y + x, [a, b])


def test_broadcasting_row_vector_against_matrix(rng):
    matrix, row = rng.normal(size=(5, 3)), rng.normal(size=(3,))
    assert_matches_torch(lambda m, r: m * r, lambda m, r: m * r, [matrix, row])


def test_broadcasting_column_vector_against_matrix(rng):
    matrix, column = rng.normal(size=(5, 3)), rng.normal(size=(5, 1))
    assert_matches_torch(lambda m, c: m + c, lambda m, c: m + c, [matrix, column])


def test_matmul(rng):
    a, b = rng.normal(size=(4, 5)), rng.normal(size=(5, 3))
    assert_matches_torch(lambda x, y: x @ y, lambda x, y: x @ y, [a, b])


def test_batched_matmul(rng):
    a, b = rng.normal(size=(2, 4, 5)), rng.normal(size=(2, 5, 3))
    assert_matches_torch(lambda x, y: x @ y, lambda x, y: x @ y, [a, b])


def test_relu(rng):
    a = rng.normal(size=(6, 4))
    assert_matches_torch(lambda x: x.relu(), lambda x: torch.relu(x), [a])


def test_tanh(rng):
    a = rng.normal(size=(6, 4))
    assert_matches_torch(lambda x: x.tanh(), lambda x: torch.tanh(x), [a])


def test_exp_and_log(rng):
    a = np.abs(rng.normal(size=(6, 4))) + 0.5
    assert_matches_torch(lambda x: x.exp().log(), lambda x: x.exp().log(), [a])


def test_power(rng):
    a = np.abs(rng.normal(size=(5, 5))) + 0.5
    assert_matches_torch(lambda x: x**3.0, lambda x: x**3.0, [a])


def test_sum_over_axis(rng):
    a = rng.normal(size=(4, 6))
    assert_matches_torch(lambda x: x.sum(axis=1), lambda x: x.sum(dim=1), [a])


def test_mean(rng):
    a = rng.normal(size=(4, 6))
    assert_matches_torch(lambda x: x.mean(), lambda x: x.mean(), [a])


def test_log_softmax(rng):
    a = rng.normal(size=(7, 5))
    assert_matches_torch(
        lambda x: x.log_softmax(axis=-1),
        lambda x: torch.log_softmax(x, dim=-1),
        [a],
    )


def test_log_softmax_stays_finite_on_extreme_logits():
    """A naive softmax overflows here; the log-sum-exp shift must not."""
    extreme = np.array([[1000.0, 1001.0, 999.0]])
    ours = Tensor(extreme, requires_grad=True)
    result = ours.log_softmax(axis=-1)
    assert np.all(np.isfinite(result.data))

    theirs = torch.tensor(extreme, dtype=torch.float64, requires_grad=True)
    np.testing.assert_allclose(result.data, torch.log_softmax(theirs, dim=-1).detach().numpy(), atol=1e-12)


def test_reshape_and_transpose(rng):
    a = rng.normal(size=(4, 6))
    assert_matches_torch(lambda x: x.reshape(6, 4), lambda x: x.reshape(6, 4), [a])
    assert_matches_torch(lambda x: x.transpose(), lambda x: x.transpose(-1, -2), [a])


def test_indexing_accumulates_repeated_rows(rng):
    """A row selected twice must receive the sum of both upstream gradients."""
    a = rng.normal(size=(5, 3))
    index = np.array([0, 2, 0, 4])
    assert_matches_torch(lambda x: x[index], lambda x: x[index], [a])


def test_full_mlp_forward_and_backward_matches_torch(rng):
    x = rng.normal(size=(16, 8))
    w1, b1 = rng.normal(size=(8, 12)), rng.normal(size=(12,))
    w2, b2 = rng.normal(size=(12, 4)), rng.normal(size=(4,))
    targets = rng.integers(0, 4, size=16)

    def ours(x_, w1_, b1_, w2_, b2_):
        hidden = (x_ @ w1_ + b1_).relu()
        logits = hidden @ w2_ + b2_
        log_probabilities = logits.log_softmax(axis=-1)
        return -log_probabilities[np.arange(16), targets].mean()

    def theirs(x_, w1_, b1_, w2_, b2_):
        hidden = torch.relu(x_ @ w1_ + b1_)
        logits = hidden @ w2_ + b2_
        return torch.nn.functional.cross_entropy(logits, torch.tensor(targets))

    assert_matches_torch(ours, theirs, [x, w1, b1, w2, b2], tolerance=1e-10)
