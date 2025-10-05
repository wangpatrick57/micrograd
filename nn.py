import random

from engine import Value


class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1.0, 1.0)) for _ in range(nin)]
        self.b = Value(random.uniform(-1.0, 1.0))

    def __call__(self, x: list[Value | float]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), start=self.b)
        out = act.tanh()
        return out

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[Value | float]) -> list[Value]:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self) -> list[Value]:
        return [
            parameter for neuron in self.neurons for parameter in neuron.parameters()
        ]


class MLP:
    def __init__(self, szs: list[int]):
        self.layers = [Layer(nin, nout) for nin, nout in zip(szs, szs[1:])]

    def __call__(self, x: list[Value | float]) -> list[Value]:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Value]:
        return [parameter for layer in self.layers for parameter in layer.parameters()]


if __name__ == "__main__":
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]
    m = MLP([3, 4, 4, 1])
    lr = 0.05

    for _ in range(1000):
        ypreds = [m(x)[0] for x in xs]
        loss: Value = sum((y - ypred) ** 2.0 for y, ypred in zip(ys, ypreds))
        print("loss", loss)

        for parameter in m.parameters():
            parameter.grad = 0.0

        loss.backward()

        for parameter in m.parameters():
            parameter.data -= lr * parameter.grad
    
    print(ypreds)
