"""Train a classifier on real handwritten digits using only this engine."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tinygrad_core import Tensor, MLP, Adam, cross_entropy


def load_data(seed: int = 0):
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=seed, stratify=digits.target
    )
    scaler = StandardScaler().fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test), y_train, y_test


def accuracy(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    logits = model(Tensor(x))
    return float((logits.data.argmax(axis=1) == y).mean())


def train(epochs: int = 60, batch_size: int = 64, seed: int = 0, verbose: bool = True):
    x_train, x_test, y_train, y_test = load_data(seed)
    model = MLP([x_train.shape[1], 64, 32, 10], seed=seed)
    optimizer = Adam(model.parameters(), learning_rate=3e-3)
    rng = np.random.default_rng(seed)

    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(x_train))
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = cross_entropy(model(Tensor(x_train[batch])), y_train[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.data) * len(batch)

        epoch_loss /= len(order)
        history.append(epoch_loss)
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1:3d}  loss {epoch_loss:.4f}  test accuracy {accuracy(model, x_test, y_test):.4f}")

    return model, history, (x_train, x_test, y_train, y_test)


def main() -> None:
    print("Training a 64-64-32-10 MLP on scikit-learn digits using this engine")
    model, history, (x_train, x_test, y_train, y_test) = train()
    print(f"\nFinal training loss: {history[-1]:.4f}")
    print(f"Train accuracy: {accuracy(model, x_train, y_train):.4f}")
    print(f"Test accuracy:  {accuracy(model, x_test, y_test):.4f}")


if __name__ == "__main__":
    main()
