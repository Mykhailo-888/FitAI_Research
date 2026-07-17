"""
FitAI Physiological Hierarchical VAE

decoder.py

Universal decoder.

Author: Mykhailo Velychko
"""

from ml.vae.layers import DenseLayer
from ml.vae.activations import LeakyReLU


class VariationalDecoder:
    """
    Universal decoder.

    BAI
      ↓
    Hidden
      ↓
    Reconstruction
    """

    def __init__(
            self,
            latent_size,
            hidden_size,
            output_size):

        self.hidden = DenseLayer(
            latent_size,
            hidden_size
        )

        self.activation = LeakyReLU()

        self.output = DenseLayer(
            hidden_size,
            output_size
        )

    def forward(self, z):

        h = self.hidden.forward(z)

        h = self.activation.forward(h)

        reconstruction = self.output.forward(h)

        return reconstruction

    def backward(self, grad):

        grad = self.output.backward(grad)

        grad = self.activation.backward(grad)

        grad = self.hidden.backward(grad)

        return grad

    def update(self, lr):

        self.hidden.update(lr)

        self.output.update(lr)

    def zero_grad(self):

        self.hidden.zero_grad()

        self.output.zero_grad()

    def save(self):

        return {

            "hidden": self.hidden.save(),

            "output": self.output.save()

        }

    def load(self, state):

        self.hidden.load(state["hidden"])

        self.output.load(state["output"])