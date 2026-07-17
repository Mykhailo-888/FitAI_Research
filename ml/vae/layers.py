"""
FitAI Physiological Hierarchical VAE

layers.py

Core neural network layers used by the entire FitAI research framework.

Author: Mykhailo Velychko
"""

import numpy as np


class BaseLayer:
    """
    Base class for all neural network layers.
    """

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def update(self, lr):
        pass

    def zero_grad(self):
        pass

    def save(self):
        return {}

    def load(self, state):
        pass


class DenseLayer(BaseLayer):
    """
    Fully connected neural layer.
    """

    def __init__(self, input_size, output_size):

        limit = np.sqrt(6 / (input_size + output_size))

        self.W = np.random.uniform(
            -limit,
            limit,
            (input_size, output_size)
        )

        self.b = np.zeros((1, output_size))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):

        self.x = x

        return x @ self.W + self.b

    def backward(self, grad):

        self.dW += self.x.T @ grad
        self.db += np.sum(grad, axis=0, keepdims=True)

        return grad @ self.W.T

    def update(self, lr):

        self.W -= lr * self.dW
        self.b -= lr * self.db

    def zero_grad(self):

        self.dW.fill(0)
        self.db.fill(0)

    def save(self):

        return {

            "W": self.W,
            "b": self.b

        }

    def load(self, state):

        self.W = state["W"]
        self.b = state["b"]


class DropoutLayer(BaseLayer):
    """
    Dropout regularization.
    """

    def __init__(self, dropout_rate=0.2):

        self.dropout_rate = dropout_rate

    def forward(self, x, training=True):

        if not training:
            return x

        self.mask = (
            np.random.rand(*x.shape)
            > self.dropout_rate
        )

        return x * self.mask / (1 - self.dropout_rate)

    def backward(self, grad):

        return grad * self.mask / (1 - self.dropout_rate)


class LayerNormalization(BaseLayer):
    """
    Layer Normalization.
    """

    def __init__(self, eps=1e-5):

        self.eps = eps

    def forward(self, x):

        self.mean = np.mean(x, axis=1, keepdims=True)

        self.var = np.var(x, axis=1, keepdims=True)

        self.std = np.sqrt(self.var + self.eps)

        return (x - self.mean) / self.std

    def backward(self, grad):

        return grad


class ResidualConnection(BaseLayer):
    """
    Residual connection.
    """

    def forward(self, x, residual):

        return x + residual

    def backward(self, grad):

        return grad


class Sequential:
    """
    Simple sequential container.
    """

    def __init__(self):

        self.layers = []

    def add(self, layer):

        self.layers.append(layer)

    def forward(self, x):

        for layer in self.layers:

            x = layer.forward(x)

        return x

    def backward(self, grad):

        for layer in reversed(self.layers):

            grad = layer.backward(grad)

        return grad

    def update(self, lr):

        for layer in self.layers:

            layer.update(lr)

    def zero_grad(self):

        for layer in self.layers:

            if hasattr(layer, "zero_grad"):

                layer.zero_grad()

    def save(self):

        return [

            layer.save()

            for layer in self.layers

        ]

    def load(self, states):

        for layer, state in zip(self.layers, states):

            layer.load(state)