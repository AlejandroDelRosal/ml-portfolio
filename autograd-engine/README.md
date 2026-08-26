# Autograd Engine

A reverse-mode automatic differentiation engine and neural network library written from scratch on NumPy, validated to machine precision against PyTorch.

The point is not to replace PyTorch. It is to implement the machinery that makes deep learning work, then prove the implementation is correct rather than assert it: every gradient is checked against both PyTorch's autograd and independent finite differences, and a full training run is compared against PyTorch epoch by epoch.

## What it implements

Reverse-mode autodiff over N-dimensional arrays. Each operation records a closure that propagates gradients to its inputs; `backward()` walks a topologically sorted graph in reverse, accumulating gradients at every node.

The parts that are genuinely easy to get wrong, and are therefore tested hardest:

- **Broadcasting-aware gradients.** When an operand is broadcast in the forward pass, each replica receives its own share of the upstream gradient, so the reverse pass must sum over exactly the axes that were expanded. Getting this wrong produces gradients of the wrong shape, or silently wrong values.
- **Numerically stable log-softmax.** Exponentiating raw logits overflows; subtracting the row maximum first cancels exactly in the log-sum-exp identity. A test asserts the result stays finite on logits of magnitude 1000, where a naive implementation returns `nan`.
- **Gradient accumulation on repeated indices.** Selecting the same row twice means that row must receive the sum of both upstream gradients, not the last one written.
- **Diamond-shaped graphs.** A tensor feeding two branches that later merge must accumulate contributions from both paths.

Built on top of the engine: `Linear` and `MLP` layers with Kaiming initialization, ReLU and tanh activations, cross-entropy and MSE losses, and SGD (with momentum) and Adam optimizers.

## Validation

**28 tests, all passing.** Three independent layers of evidence:

1. **Against PyTorch (16 tests).** Every operation is run through both frameworks on identical inputs and the resulting gradients compared, including broadcasting in both directions, batched matmul, and a complete MLP forward and backward pass. All agree to within `1e-9`.

2. **Against finite differences (6 tests).** PyTorch and this engine could in principle share a conceptual mistake. Central differences depend on nothing but the forward pass, so agreement confirms the backward pass really is the derivative of the forward pass.

3. **Against closed-form solutions (6 tests).** SGD on linear regression converges to the analytic least-squares optimum; a network learns XOR, which no linear model can represent; Adam's first step is verified to be full-sized, confirming the bias correction works.

## Head-to-head with PyTorch

Same architecture, same initial weights, same batches, same optimizer hyperparameters, 30 epochs on the scikit-learn digits dataset:

| Epoch | This engine | PyTorch | Absolute difference |
|---|---|---|---|
| 1 | 1.62153666 | 1.62153666 | 0.00e+00 |
| 5 | 0.09530724 | 0.09530724 | 0.00e+00 |
| 10 | 0.02376423 | 0.02376423 | 6.94e-18 |
| 20 | 0.00394383 | 0.00394383 | 8.67e-19 |
| 30 | 0.00159023 | 0.00159023 | 4.55e-18 |

Maximum per-epoch loss difference across the whole run: **5.55e-17**. Double-precision machine epsilon is 2.2e-16, so the two implementations are numerically indistinguishable, not merely close. Both reach identical test accuracy (0.9644).

Training the same network to convergence reaches **96.9% test accuracy** on held-out digits.

On this workload the engine also runs faster in wall-clock terms (0.39s versus 2.03s). That is not a claim of superiority: at this model size PyTorch's per-operation dispatch overhead dominates its highly optimized kernels. On large models, or on a GPU, PyTorch wins by orders of magnitude.

## Layout

```
tinygrad_core/
    tensor.py   autodiff engine: the graph, the ops and their gradients
    nn.py       layers, initialization, losses
    optim.py    SGD with momentum, Adam with bias correction
tests/          gradient checks against PyTorch, finite differences, closed forms
benchmarks/     head-to-head training run against PyTorch
examples/       training a classifier on real handwritten digits
```

## Running it

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python -m examples.train_digits
.venv/bin/python -m benchmarks.compare_with_torch
```
