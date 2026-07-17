"""
FitAI Physiological Hierarchical VAE

encoder.py

Universal variational encoder.

Author: Mykhailo Velychko
"""

from .layers import DenseLayer
from .activations import LeakyReLU
from .sampling import SamplingLayer


class VariationalEncoder:
    """
    Universal Variational Encoder

    Input
        ↓
    Dense
        ↓
    LeakyReLU
        ↓
    μ ---------
               │
    log_var ---│
               ▼
         Sampling Layer
               │
               ▼
               z
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            latent_size):

        self.hidden = DenseLayer(
            input_size,
            hidden_size
        )

        self.activation = LeakyReLU()

        self.mu_layer = DenseLayer(
            hidden_size,
            latent_size
        )

        self.log_var_layer = DenseLayer(
            hidden_size,
            latent_size
        )

        self.sampling = SamplingLayer()

    # =====================================================
    # Forward
    # =====================================================

    def forward(self, x):

        h = self.hidden.forward(x)

        h = self.activation.forward(h)

        mu = self.mu_layer.forward(h)

        log_var = self.log_var_layer.forward(h)

        z = self.sampling.forward(mu, log_var)

        return mu, log_var, z

    # =====================================================
    # Backward
    # =====================================================

    def backward(self, grad):

        grad_mu, grad_log_var = self.sampling.backward(grad)

        grad_mu = self.mu_layer.backward(
            grad_mu
        )

        grad_log_var = self.log_var_layer.backward(
            grad_log_var
        )

        grad_hidden = grad_mu + grad_log_var

        grad_hidden = self.activation.backward(
            grad_hidden
        )

        grad_input = self.hidden.backward(
            grad_hidden
        )

        return grad_input

    # =====================================================
    # Update
    # =====================================================

    def update(self, lr):

        self.hidden.update(lr)

        self.mu_layer.update(lr)

        self.log_var_layer.update(lr)

    # =====================================================
    # Zero gradients
    # =====================================================

    def zero_grad(self):

        self.hidden.zero_grad()

        self.mu_layer.zero_grad()

        self.log_var_layer.zero_grad()

    # =====================================================
    # Save
    # =====================================================

    def save(self):

        return {

            "hidden": self.hidden.save(),

            "mu_layer": self.mu_layer.save(),

            "log_var_layer": self.log_var_layer.save()

        }

    # =====================================================
    # Load
    # =====================================================

    def load(self, state):

        self.hidden.load(
            state["hidden"]
        )

        self.mu_layer.load(
            state["mu_layer"]
        )

        self.log_var_layer.load(
            state["log_var_layer"]
        )