"""
FitAI Physiological Hierarchical VAE

bai.py

Bioenergetic Adaptation Index (BAI)

Author: Mykhailo Velychko
"""

import numpy as np

from ml.vae.layers import DenseLayer
from ml.vae.activations import LeakyReLU
from ml.vae.sampling import SamplingLayer


class BioenergeticCore:
    """
    Global physiological latent representation.

    Energy
    Recovery
    Stress
    Muscle
    Metabolism
    Aging
            │
            ▼
    Bioenergetic Core
            │
            ▼
          μ , log_var
            │
            ▼
         Sampling
            │
            ▼
           BAI
    """

    def __init__(
            self,
            input_size=24,
            hidden_size=16,
            latent_size=4):

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

    def forward(self, latent_states):

        latent_vector = np.concatenate(

            [

                latent_states["energy"],
                latent_states["recovery"],
                latent_states["stress"],
                latent_states["muscle"],
                latent_states["metabolism"],
                latent_states["aging"]

            ],

            axis=1

        )

        h = self.hidden.forward(latent_vector)

        h = self.activation.forward(h)

        mu = self.mu_layer.forward(h)

        log_var = self.log_var_layer.forward(h)

        bai = self.sampling.forward(mu, log_var)

        return {

            "latent_vector": latent_vector,

            "mu": mu,

            "log_var": log_var,

            "bai": bai

        }

    # =====================================================
    # Backward
    # =====================================================

    # =====================================================
    # Backward
    # =====================================================

    def backward(self, grad):
        grad_mu, grad_log_var = self.sampling.backward(
            grad
        )

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

        # --------------------------------------------------
        # Split back into 6 latent spaces
        # --------------------------------------------------

        return {

            "energy": grad_input[:, 0:4],

            "recovery": grad_input[:, 4:8],

            "stress": grad_input[:, 8:12],

            "muscle": grad_input[:, 12:16],

            "metabolism": grad_input[:, 16:20],

            "aging": grad_input[:, 20:24]

        }
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