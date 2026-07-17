"""
FitAI Physiological Hierarchical VAE

sampling.py

Reparameterization layer used in Variational Autoencoder.

Author: Mykhailo Velychko
"""

import numpy as np

from .layers import BaseLayer


class SamplingLayer(BaseLayer):
    """
    Reparameterization Trick

    z = μ + σ * ε

    ε ~ N(0, I)
    """

    def __init__(self, random_seed=None):

        if random_seed is not None:
            np.random.seed(random_seed)

        self.mu = None
        self.log_var = None
        self.std = None
        self.epsilon = None

    def forward(self, mu, log_var):
        """
        Parameters
        ----------
        mu : ndarray
            Mean vector.

        log_var : ndarray
            Log variance vector.

        Returns
        -------
        ndarray
            Latent vector z.
        """

        self.mu = mu
        self.log_var = log_var

        # σ = exp(0.5 * log(σ²))
        self.std = np.exp(0.5 * log_var)

        # ε ~ N(0,1)
        self.epsilon = np.random.randn(*mu.shape)

        # Reparameterization
        z = mu + self.std * self.epsilon

        return z

    def backward(self, grad):
        """
        Backpropagation through sampling.

        Returns
        -------
        grad_mu
        grad_log_var
        """

        grad_mu = grad

        grad_log_var = (
            grad
            * self.epsilon
            * self.std
            * 0.5
        )

        return grad_mu, grad_log_var

    def update(self, lr):
        """
        Sampling layer has no trainable parameters.
        """
        pass