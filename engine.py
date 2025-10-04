import math


class Value:
    def __init__(self, data: float):
        self.data = data
        self.grad: float | None = None
        self._prev: tuple["Value", ...] | None = None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: "Value") -> "Value":
        return Value(self.data + other.data, (self, other))

    def __mul__(self, other: "Value") -> "Value":
        return Value(self.data * other.data, (self, other))

    def tanh(self) -> "Value":
        return Value(
            (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1),
            (self,),
        )


if __name__ == "__main__":
    a = Value(2.0)
    print(a.tanh())
