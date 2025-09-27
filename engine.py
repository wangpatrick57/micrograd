from enum import Enum


class Op(Enum):
    NONE = ""
    ADD = "+"
    MUL = "*"


class Value:
    def __init__(self, data: float, _prev: list["Value"] = [], _op: Op = Op.NONE):
        self.data = data
        self._prev = _prev
        self._op = _op

    def __repr__(self) -> str:
        return f"Value(data={self.data}, _prev={self._prev}, _op={self._op})"

    def __add__(self, other: "Value") -> "Value":
        return Value(self.data + other.data, [self, other], Op.ADD)

    def __mul__(self, other: "Value") -> "Value":
        return Value(self.data * other.data, [self, other], Op.MUL)


if __name__ == "__main__":
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    print(a * b + c)
