import numpy as np


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

class Sigmoid:
    def __init__(self):
        self.params = []

    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self):
        self.params = []

    def forward(self, x):
        W, b = self.params
        return np.dot(x, W) + b


class TwolayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # レイア
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

