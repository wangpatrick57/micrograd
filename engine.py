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
        a = self.data
        b = other.data
        out = a + b

        def grad_fn(out_grad: float) -> tuple[float, float]:
            a_grad = 1 * out_grad
            b_grad = 1 * out_grad
            return (a_grad, b_grad)

        return Value(out, _prev=(self, other), _grad_fn=grad_fn)

    def __mul__(self, other: "Value") -> "Value":
        a = self.data
        b = other.data
        out = a * b

        def grad_fn(out_grad: float) -> tuple[float, float]:
            a_grad = b * out_grad
            b_grad = a * out_grad
            return (a_grad, b_grad)

        return Value(out, _prev=(self, other), _grad_fn=grad_fn)

    def tanh(self) -> "Value":
        return Value(
            (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1),
            _prev=(self,),
        )

    def _is_leaf(self) -> bool:
        if self._prev is None:
            assert self._grad_fn is None
            return True
        else:
            assert self._grad_fn is not None
            return False

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
    b = Value(3.0)
    c = a * b
    c.backward()
    print(a.grad, b.grad, c.grad)
