"""
FitAI Physiological Hierarchical VAE

model.py

Main Hierarchical VAE model.

Author: Mykhailo Velychko
"""

from ml.bioenergetics.latent_states import HierarchicalLatentStates
from ml.bioenergetics.bai import BioenergeticCore
from ml.vae.decoder import VariationalDecoder


class FitAIVAE:
    """
    Complete FitAI Hierarchical β-VAE

    Input: 23 physiological features

            23 features
                  │
                  ▼
        Hierarchical Latent States
        6 local physiological encoders
                  │
                  ▼
        Concatenated local latent states
                  │
                  ▼
        Bioenergetic Core (BAI)
                  │
                  ▼
           Global Decoder
                  │
                  ▼
        Reconstructed 23 features

    Local reconstruction path:

        Local latent state
                │
                ▼
          Local Decoder
                │
                ▼
      Reconstructed subsystem features
    """

    def __init__(self):

        self.latent_states = HierarchicalLatentStates()

        self.bioenergetic_core = BioenergeticCore()

        self.decoder = VariationalDecoder(
            latent_size=4,
            hidden_size=16,
            output_size=23
        )

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(self, x):
        """
        Full forward pass.

        Parameters
        ----------
        x : np.ndarray
            Normalized input matrix with shape:

            (batch_size, 23)

        Returns
        -------
        dict
            latent_states:
                outputs of local physiological encoders
                and local decoders

            bai:
                output of the Bioenergetic Core

            reconstruction:
                global reconstruction of all 23 features
        """

        latent = self.latent_states.forward(x)

        bai = self.bioenergetic_core.forward(latent)

        reconstruction = self.decoder.forward(
            bai["bai"]
        )

        return {
            "latent_states": latent,
            "bai": bai,
            "reconstruction": reconstruction
        }

    # ==========================================================
    # Backward
    # ==========================================================

    def backward(self, grads):
        """
        Full backward pass.

        Expected gradient dictionary:

        grads["global"]
            Gradient of the global reconstruction loss.
            Shape: (batch_size, 23)

        grads["energy"]
            Shape: (batch_size, 6)

        grads["recovery"]
            Shape: (batch_size, 5)

        grads["stress"]
            Shape: (batch_size, 5)

        grads["muscle"]
            Shape: (batch_size, 6)

        grads["metabolism"]
            Shape: (batch_size, 5)

        grads["aging"]
            Shape: (batch_size, 5)
        """

        required_keys = {
            "global",
            "energy",
            "recovery",
            "stress",
            "muscle",
            "metabolism",
            "aging"
        }

        missing_keys = required_keys.difference(
            grads.keys()
        )

        if missing_keys:
            raise KeyError(
                "Missing gradients in FitAIVAE.backward(): "
                f"{sorted(missing_keys)}"
            )

        # ------------------------------------------------------
        # 1. Global reconstruction path
        #
        # global reconstruction loss
        #       ↓
        # global decoder
        #       ↓
        # BAI latent state
        # ------------------------------------------------------

        grad_bai = self.decoder.backward(
            grads["global"]
        )

        # ------------------------------------------------------
        # 2. Bioenergetic Core
        #
        # BAI latent gradient
        #       ↓
        # concatenated local latent states
        # ------------------------------------------------------

        latent_grads = self.bioenergetic_core.backward(
            grad_bai
        )

        # ------------------------------------------------------
        # 3. Global gradient path to local encoders
        #
        # concatenated latent gradients
        #       ↓
        # six local encoders
        # ------------------------------------------------------

        self.latent_states.backward_global(
            latent_grads
        )

        # ------------------------------------------------------
        # 4. Local reconstruction paths
        #
        # local reconstruction losses
        #       ↓
        # six local decoders
        #       ↓
        # six local encoders
        # ------------------------------------------------------

        self.latent_states.backward_local(
            grads
        )

    # ==========================================================
    # Update
    # ==========================================================

    def update(self, lr):
        """
        Update all trainable model parameters.
        """

        if lr <= 0:
            raise ValueError(
                "Learning rate must be greater than zero"
            )

        self.latent_states.update(lr)

        self.bioenergetic_core.update(lr)

        self.decoder.update(lr)

    # ==========================================================
    # Zero gradients
    # ==========================================================

    def zero_grad(self):
        """
        Reset accumulated gradients before a new epoch.
        """

        self.latent_states.zero_grad()

        self.bioenergetic_core.zero_grad()

        self.decoder.zero_grad()

    # ==========================================================
    # Save
    # ==========================================================

    def save(self):
        """
        Return serializable model state.
        """

        return {
            "latent_states": self.latent_states.save(),
            "bioenergetic_core": (
                self.bioenergetic_core.save()
            ),
            "decoder": self.decoder.save()
        }

    # ==========================================================
    # Load
    # ==========================================================

    def load(self, state):
        """
        Restore model state.
        """

        required_keys = {
            "latent_states",
            "bioenergetic_core",
            "decoder"
        }

        missing_keys = required_keys.difference(
            state.keys()
        )

        if missing_keys:
            raise KeyError(
                "Missing model state keys: "
                f"{sorted(missing_keys)}"
            )

        self.latent_states.load(
            state["latent_states"]
        )

        self.bioenergetic_core.load(
            state["bioenergetic_core"]
        )

        self.decoder.load(
            state["decoder"]
        )