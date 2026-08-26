"""Head-to-head against PyTorch: identical architecture, initialization and data.

If the engine is correct, both frameworks fed the same weights and the same
batches must trace the same loss curve, not merely reach a similar accuracy.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from tinygrad_core import Tensor, MLP, Adam, cross_entropy
from examples.train_digits import load_data

LAYER_SIZES = [64, 64, 32, 10]
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
SEED = 0


def build_torch_model_from(ours: MLP) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    for position, layer in enumerate(ours.layers):
        linear = torch.nn.Linear(layer.weight.shape[0], layer.weight.shape[1]).double()
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(layer.weight.data.T))
            linear.bias.copy_(torch.tensor(layer.bias.data))
        layers.append(linear)
        if position < len(ours.layers) - 1:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def run_ours(model, x_train, y_train, batches) -> list[float]:
    optimizer = Adam(model.parameters(), learning_rate=LEARNING_RATE)
    losses = []
    for epoch_batches in batches:
        total = 0.0
        for batch in epoch_batches:
            optimizer.zero_grad()
            loss = cross_entropy(model(Tensor(x_train[batch])), y_train[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.data) * len(batch)
        losses.append(total / sum(len(b) for b in epoch_batches))
    return losses


def run_torch(model, x_train, y_train, batches) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x_tensor = torch.tensor(x_train, dtype=torch.float64)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    losses = []
    for epoch_batches in batches:
        total = 0.0
        for batch in epoch_batches:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x_tensor[batch]), y_tensor[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        losses.append(total / sum(len(b) for b in epoch_batches))
    return losses


def main() -> None:
    x_train, x_test, y_train, y_test = load_data(SEED)

    rng = np.random.default_rng(SEED)
    batches = [
        [order[start : start + BATCH_SIZE] for start in range(0, len(order), BATCH_SIZE)]
        for order in (rng.permutation(len(x_train)) for _ in range(EPOCHS))
    ]

    ours = MLP(LAYER_SIZES, seed=SEED)
    theirs = build_torch_model_from(ours)

    start = time.perf_counter()
    our_losses = run_ours(ours, x_train, y_train, batches)
    our_time = time.perf_counter() - start

    start = time.perf_counter()
    their_losses = run_torch(theirs, x_train, y_train, batches)
    their_time = time.perf_counter() - start

    our_accuracy = float((ours(Tensor(x_test)).data.argmax(axis=1) == y_test).mean())
    with torch.no_grad():
        their_predictions = theirs(torch.tensor(x_test, dtype=torch.float64)).argmax(dim=1).numpy()
    their_accuracy = float((their_predictions == y_test).mean())

    differences = np.abs(np.array(our_losses) - np.array(their_losses))

    print(f"{'epoch':>6}  {'this engine':>12}  {'pytorch':>12}  {'abs diff':>10}")
    for epoch in [0, 4, 9, 19, EPOCHS - 1]:
        print(f"{epoch + 1:>6}  {our_losses[epoch]:>12.8f}  {their_losses[epoch]:>12.8f}  {differences[epoch]:>10.2e}")

    print(f"\nMaximum per-epoch loss difference over {EPOCHS} epochs: {differences.max():.3e}")
    print(f"Test accuracy: this engine {our_accuracy:.4f}, pytorch {their_accuracy:.4f}")
    print(f"Wall-clock training time: this engine {our_time:.2f}s, pytorch {their_time:.2f}s")


if __name__ == "__main__":
    main()
