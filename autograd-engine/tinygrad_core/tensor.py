from __future__ import annotations

import numpy as np


def _unbroadcast(gradient: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Reduce a gradient back to the shape of the operand that produced it.

    Broadcasting during the forward pass replicates an operand across new or
    size-one axes. Each replica receives its own share of the upstream
    gradient, so the reverse pass must sum over exactly those axes.
    """
    while gradient.ndim > len(target_shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(target_shape)


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_parents")

    def __init__(self, data, requires_grad: bool = False, _parents: tuple = ()):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._backward = lambda: None
        self._parents = _parents

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def _child(self, data, parents) -> Tensor:
        requires_grad = any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad=requires_grad, _parents=tuple(parents) if requires_grad else ())

    def _accumulate(self, gradient: np.ndarray) -> None:
        if not self.requires_grad:
            return
        reduced = _unbroadcast(gradient, self.shape)
        self.grad = reduced if self.grad is None else self.grad + reduced

    def backward(self) -> None:
        if self.data.size != 1:
            raise RuntimeError("backward() requires a scalar output; reduce with sum() or mean() first")

        ordered: list[Tensor] = []
        visited: set[int] = set()

        def visit(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for parent in node._parents:
                visit(parent)
            ordered.append(node)

        visit(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(ordered):
            # A node reachable from the output can still be one that no
            # gradient flows into, for example a constant target combined with
            # a trainable prediction. Such a node never receives an upstream
            # gradient, so its backward closure has nothing to propagate.
            if node.grad is not None:
                node._backward()

    def zero_grad(self) -> None:
        self.grad = None

    def __add__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data + other.data, (self, other))

        def backward() -> None:
            self._accumulate(out.grad)
            other._accumulate(out.grad)

        out._backward = backward
        return out

    def __mul__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = self._child(self.data * other.data, (self, other))

        def backward() -> None:
            self._accumulate(out.grad * other.data)
            other._accumulate(out.grad * self.data)

        out._backward = backward
        return out

    def __pow__(self, exponent: float) -> Tensor:
        out = self._child(self.data**exponent, (self,))

        def backward() -> None:
            self._accumulate(out.grad * exponent * self.data ** (exponent - 1))

        out._backward = backward
        return out

    def matmul(self, other: Tensor) -> Tensor:
        out = self._child(self.data @ other.data, (self, other))

        def backward() -> None:
            self._accumulate(out.grad @ np.swapaxes(other.data, -1, -2))
            other._accumulate(np.swapaxes(self.data, -1, -2) @ out.grad)

        out._backward = backward
        return out

    def sum(self, axis=None, keepdims: bool = False) -> Tensor:
        out = self._child(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def backward() -> None:
            gradient = out.grad
            if axis is not None and not keepdims:
                gradient = np.expand_dims(gradient, axis)
            self._accumulate(np.broadcast_to(gradient, self.shape))

        out._backward = backward
        return out

    def mean(self, axis=None, keepdims: bool = False) -> Tensor:
        divisor = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / divisor)

    def relu(self) -> Tensor:
        out = self._child(np.maximum(self.data, 0), (self,))

        def backward() -> None:
            self._accumulate(out.grad * (self.data > 0))

        out._backward = backward
        return out

    def tanh(self) -> Tensor:
        activated = np.tanh(self.data)
        out = self._child(activated, (self,))

        def backward() -> None:
            self._accumulate(out.grad * (1 - activated**2))

        out._backward = backward
        return out

    def exp(self) -> Tensor:
        activated = np.exp(self.data)
        out = self._child(activated, (self,))

        def backward() -> None:
            self._accumulate(out.grad * activated)

        out._backward = backward
        return out

    def log(self) -> Tensor:
        out = self._child(np.log(self.data), (self,))

        def backward() -> None:
            self._accumulate(out.grad / self.data)

        out._backward = backward
        return out

    def log_softmax(self, axis: int = -1) -> Tensor:
        """Numerically stable log-softmax.

        Subtracting the row maximum before exponentiating keeps exp() away
        from overflow; the shift cancels exactly in the log-sum-exp identity,
        so the result is unchanged.
        """
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        log_sum_exp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        result = shifted - log_sum_exp
        out = self._child(result, (self,))

        def backward() -> None:
            softmax = np.exp(result)
            self._accumulate(out.grad - softmax * out.grad.sum(axis=axis, keepdims=True))

        out._backward = backward
        return out

    def reshape(self, *shape: int) -> Tensor:
        original_shape = self.shape
        out = self._child(self.data.reshape(shape), (self,))

        def backward() -> None:
            self._accumulate(out.grad.reshape(original_shape))

        out._backward = backward
        return out

    def transpose(self) -> Tensor:
        out = self._child(np.swapaxes(self.data, -1, -2), (self,))

        def backward() -> None:
            self._accumulate(np.swapaxes(out.grad, -1, -2))

        out._backward = backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __truediv__(self, other) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other**-1.0

    def __radd__(self, other) -> Tensor:
        return self + other

    def __rmul__(self, other) -> Tensor:
        return self * other

    def __rsub__(self, other) -> Tensor:
        return (-self) + other

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    def __getitem__(self, index) -> Tensor:
        out = self._child(self.data[index], (self,))

        def backward() -> None:
            gradient = np.zeros_like(self.data)
            np.add.at(gradient, index, out.grad)
            self._accumulate(gradient)

        out._backward = backward
        return out
