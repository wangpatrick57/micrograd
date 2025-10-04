import math
from typing import Callable


class Value:
    def __init__(
        self,
        data: float,
        _prev: tuple["Value", ...] | None = None,
        # Takes in out_grad (self.grad) and returns the gradients of each of the Values
        # in self._prev. Is a closure and can access the original inputs and outputs of
        # the forward pass.
        _grad_fn: Callable[[float], tuple[float, ...]] | None = None,
    ):
        self.data = data
        self.grad: float | None = None
        self._prev: tuple["Value", ...] | None = _prev
        self._grad_fn: Callable | None = _grad_fn

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: "Value") -> "Value":
        out_data = self.data + other.data

        def grad_fn(out_grad: float) -> tuple[float, float]:
            return (out_grad, out_grad)

        return Value(out_data, _prev=(self, other), _grad_fn=grad_fn)

    def __mul__(self, other: "Value") -> "Value":
        out_data = self.data * other.data

        def grad_fn(out_grad: float) -> tuple[float, float]:
            return (other.data * out_grad, self.data * out_grad)

        return Value(out_data, _prev=(self, other), _grad_fn=grad_fn)

    def tanh(self) -> "Value":
        out_data = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)

        def grad_fn(out_grad: float) -> tuple[float]:
            return ((1 - out_data**2) * out_grad,)

        return Value(out_data, _prev=(self,), _grad_fn=grad_fn)

    def _is_leaf(self) -> bool:
        assert (self._prev is None) == (self._grad_fn is None)
        return self._prev is None

    def _propagate(self) -> None:
        if not self._is_leaf():
            grads = self._grad_fn(self.grad)

            for p, grad in zip(self._prev, grads):
                p.grad = grad

            for p in self._prev:
                p._propagate()

    def backward(self) -> None:
        self.grad = 1.0
        self._propagate()


if __name__ == "__main__":
    a = Value(2.0)
    b = a.tanh()
    b.backward()
    print(a.grad, b.grad)
