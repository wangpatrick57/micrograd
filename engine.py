import math
from enum import Enum


class Op(Enum):
    NONE = ""
    ADD = "+"
    MUL = "*"
    TANH = "tanh"


class Value:
    def __init__(self, data: float, _prev: tuple["Value", ...] = [], _op: Op = Op.NONE):
        self.data = data
        self.grad: float | None = None
        self._prev = _prev
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: "Value") -> "Value":
        return Value(self.data + other.data, (self, other), Op.ADD)

    def __mul__(self, other: "Value") -> "Value":
        return Value(self.data * other.data, (self, other), Op.MUL)

    def tanh(self) -> "Value":
        return Value(
            (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1),
            (self,),
            Op.TANH,
        )


if __name__ == "__main__":
    a = Value(2.0)
    print(a.tanh())
