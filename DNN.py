import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []..

    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        return np.dot(x, W) + b


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        t1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        t2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # レイア
        self.layers = [
            Affine(t1, b1),
            Sigmoid(),
            Affine(t2, b2)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(s)
