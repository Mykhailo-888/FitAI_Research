"""
FitAI Physiological Hierarchical VAE

local_decoders.py

Local decoders for physiological latent states.
"""

from ml.vae.decoder import VariationalDecoder


class LocalDecoder:

    def __init__(self,
                 latent_size,
                 hidden_size,
                 output_size):

        self.decoder = VariationalDecoder(
            latent_size=latent_size,
            hidden_size=hidden_size,
            output_size=output_size
        )

    def forward(self, z):

        return self.decoder.forward(z)

    def backward(self, grad):

        return self.decoder.backward(grad)

    def update(self, lr):

        self.decoder.update(lr)

    def zero_grad(self):

        self.decoder.zero_grad()

    def save(self):

        return self.decoder.save()

    def load(self, state):

        self.decoder.load(state)