import numpy as np
"""
A collection of activation functions used in deep learning algorithms.
"""


class Linear:
    """
    Identity function and its derivative.
    """

    def __call__(self, z):
        return z

    def D(self, z):
        return 1


class Sigmoid:
    """
    Sigmoid function and its derivative.
    """

    def __call__(self, z):
        return 1 / (1 + np.exp(-z))

    def D(self, z):
        return self(z) * (1 - self(z))


class SoftMax:
    """
    Softmax function.
    """

    def __call__(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


class ReLU:
    """
    ReLU function and its derivative.
    """

    def __call__(self, z):
        return z * (z > 0)

    def D(self, z):
        return z > 0
