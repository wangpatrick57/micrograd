import math
from typing import Callable


class Value:
    def __init__(
        self,
        data: float,
        _prev: tuple["Value", ...] = (),
        # Takes in out_grad (self.grad) and returns the gradients of each of the Values
        # in self._prev. Is a closure and can access the original inputs and outputs of
        # the forward pass.
        _grad_fn: Callable[[float], tuple[float, ...]] = lambda out_grad: (),
    ):
        self.data = data
        self.grad: float = 0.0
        self._prev = _prev
        self._grad_fn = _grad_fn

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: "Value | float") -> "Value":
        if isinstance(other, float):
            other = Value(other)

        out_data = self.data + other.data

        def grad_fn(out_grad: float) -> tuple[float, float]:
            return (out_grad, out_grad)

        return Value(out_data, _prev=(self, other), _grad_fn=grad_fn)

    def __radd__(self, other: float) -> "Value":
        return Value(other).__add__(self)

    def __neg__(self) -> "Value":
        out_data = -self.data

        def grad_fn(out_grad: float) -> tuple[float]:
            return (-out_grad,)

        return Value(out_data, _prev=(self,), _grad_fn=grad_fn)

    def __sub__(self, other: "Value | float") -> "Value":
        return self + -other

    def __rsub__(self, other: float) -> "Value":
        return Value(other).__sub__(self)

    def __mul__(self, other: "Value | float") -> "Value":
        if isinstance(other, float):
            other = Value(other)

        out_data = self.data * other.data

        def grad_fn(out_grad: float) -> tuple[float, float]:
            return (other.data * out_grad, self.data * out_grad)

        return Value(out_data, _prev=(self, other), _grad_fn=grad_fn)

    def __rmul__(self, other: float) -> "Value":
        return Value(other).__mul__(self)

    def __pow__(self, other: "Value | float") -> "Value":
        if isinstance(other, float):
            other = Value(other)

        out_data = self.data**other.data

        def grad_fn(out_grad: float) -> tuple[float, float]:
            return (
                other.data * self.data ** (other.data - 1.0) * out_grad,
                math.log(self.data) * self.data**other.data * out_grad,
            )

        return Value(out_data, _prev=(self, other), _grad_fn=grad_fn)

    def __rpow__(self, other: float) -> "Value":
        return Value(other).__pow__(self)

    def __truediv__(self, other: "Value | float") -> "Value":
        return self * other**-1.0

    def __rtruediv__(self, other: float) -> "Value":
        return Value(other).__truediv__(self)

    def tanh(self) -> "Value":
        out_data = (math.exp(2.0 * self.data) - 1.0) / (math.exp(2.0 * self.data) + 1.0)

        def grad_fn(out_grad: float) -> tuple[float]:
            return ((1.0 - out_data**2.0) * out_grad,)

        return Value(out_data, _prev=(self,), _grad_fn=grad_fn)

    def exp(self) -> "Value":
        out_data = math.exp(self.data)

        def grad_fn(out_grad: float) -> float:
            return (out_data * out_grad,)

        return Value(out_data, _prev=(self,), _grad_fn=grad_fn)

    def _topo_sort(self) -> list["Value"]:
        """
        DFS-based topological sort assuming no cycles.
        """
        reversed_topo: list["Value"] = []
        visited: set["Value"] = set()

        def build_reversed_topo(curr: "Value") -> None:
            if curr not in visited:
                visited.add(curr)
                for p in curr._prev:
                    build_reversed_topo(p)
                reversed_topo.append(curr)

        build_reversed_topo(self)
        return list(reversed(reversed_topo))

    def backward(self) -> None:
        self.grad = 1.0
        for v in self._topo_sort():
            prev_grads = v._grad_fn(v.grad)
            for p, grad in zip(v._prev, prev_grads):
                p.grad += grad


if __name__ == "__main__":
    a = Value(3.0)
    b = 2.0 - a
    b.backward()
    print(a.grad, b.grad)
