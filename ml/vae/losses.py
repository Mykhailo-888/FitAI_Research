"""
FitAI Physiological Hierarchical VAE

losses.py

Loss functions for Hierarchical β-VAE.

Author: Mykhailo Velychko
"""
from ml.bioenergetics.latent_states import (
    feature_indices,
    ENERGY_FEATURES,
    RECOVERY_FEATURES,
    STRESS_FEATURES,
    MUSCLE_FEATURES,
    METABOLISM_FEATURES,
    AGING_FEATURES,
)
import numpy as np


class VAELoss:

    """
    Total β-VAE loss.

    L = Reconstruction + β * KL
    """

    def __init__(self, beta=1.0):

        self.beta = beta

    # --------------------------------------------------------

    @staticmethod
    def reconstruction_loss(x_true, x_pred):
        """
        Mean Squared Error.
        """

        return np.mean((x_true - x_pred) ** 2)

    # --------------------------------------------------------

    @staticmethod
    def kl_divergence(mu, log_var):
        """
        KL divergence between

        q(z|x)

        and

        N(0,I)

        Formula

        KL = -0.5 * Σ(
                1
                + log_var
                - μ²
                - exp(log_var)
              )
        """

        kl = -0.5 * np.sum(

            1
            + log_var
            - np.square(mu)
            - np.exp(log_var),

            axis=1

        )

        return np.mean(kl)

    # --------------------------------------------------------

    def total_loss(
            self,
            x_true,
            x_pred,
            latent):
        # -----------------------------
        # Global reconstruction
        # -----------------------------

        global_loss = self.reconstruction_loss(
            x_true,
            x_pred
        )

        # -----------------------------
        # Local reconstructions
        # -----------------------------

        energy_loss = self.reconstruction_loss(
            x_true[:, feature_indices(ENERGY_FEATURES)],
            latent["energy_reconstruction"]
        )

        recovery_loss = self.reconstruction_loss(
            x_true[:, feature_indices(RECOVERY_FEATURES)],
            latent["recovery_reconstruction"]
        )

        stress_loss = self.reconstruction_loss(
            x_true[:, feature_indices(STRESS_FEATURES)],
            latent["stress_reconstruction"]
        )

        muscle_loss = self.reconstruction_loss(
            x_true[:, feature_indices(MUSCLE_FEATURES)],
            latent["muscle_reconstruction"]
        )

        metabolism_loss = self.reconstruction_loss(
            x_true[:, feature_indices(METABOLISM_FEATURES)],
            latent["metabolism_reconstruction"]
        )

        aging_loss = self.reconstruction_loss(
            x_true[:, feature_indices(AGING_FEATURES)],
            latent["aging_reconstruction"]
        )

        # -----------------------------
        # KL
        # -----------------------------

        kl = 0

        for name in [
            "energy",
            "recovery",
            "stress",
            "muscle",
            "metabolism",
            "aging"
        ]:
            kl += self.kl_divergence(
                latent[f"{name}_mu"],
                latent[f"{name}_log_var"]
            )

        kl /= 6

        total = (
                global_loss
                + energy_loss
                + recovery_loss
                + stress_loss
                + muscle_loss
                + metabolism_loss
                + aging_loss
                + self.beta * kl
        )

        return {

            "total_loss": total,

            "global_loss": global_loss,

            "energy_loss": energy_loss,
            "recovery_loss": recovery_loss,
            "stress_loss": stress_loss,
            "muscle_loss": muscle_loss,
            "metabolism_loss": metabolism_loss,
            "aging_loss": aging_loss,

            "kl_loss": kl

        }

    def backward(self, x_true, x_pred, latent):
        batch_size = x_true.shape[0]

        grads = {}

        # ---------------------------------------------------
        # Global reconstruction
        # ---------------------------------------------------

        grads["global"] = 2.0 * (x_pred - x_true)
        grads["global"] /= batch_size

        # ---------------------------------------------------
        # Energy
        # ---------------------------------------------------

        grads["energy"] = (
                                  2.0 * (
                                  latent["energy_reconstruction"]
                                  - x_true[:, feature_indices(ENERGY_FEATURES)]
                          )
                          ) / batch_size

        # ---------------------------------------------------

        grads["recovery"] = (
                                    2.0 * (
                                    latent["recovery_reconstruction"]
                                    - x_true[:, feature_indices(RECOVERY_FEATURES)]
                            )
                            ) / batch_size

        # ---------------------------------------------------

        grads["stress"] = (
                                  2.0 * (
                                  latent["stress_reconstruction"]
                                  - x_true[:, feature_indices(STRESS_FEATURES)]
                          )
                          ) / batch_size

        # ---------------------------------------------------

        grads["muscle"] = (
                                  2.0 * (
                                  latent["muscle_reconstruction"]
                                  - x_true[:, feature_indices(MUSCLE_FEATURES)]
                          )
                          ) / batch_size

        # ---------------------------------------------------

        grads["metabolism"] = (
                                      2.0 * (
                                      latent["metabolism_reconstruction"]
                                      - x_true[:, feature_indices(METABOLISM_FEATURES)]
                              )
                              ) / batch_size

        # ---------------------------------------------------

        grads["aging"] = (
                                 2.0 * (
                                 latent["aging_reconstruction"]
                                 - x_true[:, feature_indices(AGING_FEATURES)]
                         )
                         ) / batch_size

        return grads