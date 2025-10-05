import random

from engine import Value


class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(random.uniform(-1.0, 1.0))

    def __call__(self, x: list[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        out = act.tanh()
        return out


class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [ni(x) for ni in self.neurons]


if __name__ == "__main__":
    x = [1.0, 3.0]
    l = Layer(2, 3)
    print(l(x))
