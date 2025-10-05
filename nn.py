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


class MLP:
    def __init__(self, szs: list[int]):
        self.layers = [Layer(nin, nout) for nin, nout in zip(szs, szs[1:])]

    def __call__(self, x: list[Value]) -> list[Value]:
        for li in self.layers:
            x = li(x)
        return x


if __name__ == "__main__":
    x = [1.0, 3.0]
    m = MLP([2, 3, 5])
    print(m(x))
