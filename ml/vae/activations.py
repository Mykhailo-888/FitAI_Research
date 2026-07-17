"""
FitAI Physiological Hierarchical VAE

activations.py

Activation functions for the FitAI neural network library.

Author: Mykhailo Velychko
"""

import numpy as np


class BaseActivation:
    """
    Base activation class.
    """

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class ReLU(BaseActivation):
    """
    Rectified Linear Unit
    """

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.x > 0)


class LeakyReLU(BaseActivation):
    """
    Leaky ReLU
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad):
        dx = np.ones_like(self.x)
        dx[self.x < 0] = self.alpha
        return grad * dx


class Sigmoid(BaseActivation):
    """
    Sigmoid activation
    """

    def forward(self, x):
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, grad):
        return grad * self.y * (1.0 - self.y)


class Tanh(BaseActivation):
    """
    Hyperbolic tangent
    """

    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad):
        return grad * (1.0 - self.y ** 2)


class Softplus(BaseActivation):
    """
    Softplus activation

    Used for sigma in VAE.
    Guarantees positive output.
    """

    def forward(self, x):
        self.x = x
        return np.log1p(np.exp(x))

    def backward(self, grad):
        sigmoid = 1.0 / (1.0 + np.exp(-self.x))
        return grad * sigmoid


class Identity(BaseActivation):
    """
    Identity activation
    """

    def forward(self, x):
        return x

    def backward(self, grad):
        return grad