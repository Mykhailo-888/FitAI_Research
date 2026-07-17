"""
FitAI Physiological Hierarchical VAE

trainer.py

Training pipeline for Hierarchical VAE.

Author: Mykhailo Velychko
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.vae.model import FitAIVAE
from ml.vae.losses import VAELoss


class VAETrainer:

    def __init__(
        self,
        csv_path,
        learning_rate=1e-3,
        epochs=200,
        beta=1.0
    ):
        self.csv_path = Path(csv_path)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.beta = beta

        self.model = FitAIVAE()
        self.loss_fn = VAELoss(beta=beta)

        self.mean = None
        self.std = None

    # --------------------------------------------------------

    def load_dataset(self):

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.csv_path}"
            )

        df = pd.read_csv(self.csv_path)

        X = df.values.astype(np.float64)

        if X.ndim != 2:
            raise ValueError(
                f"Dataset must be 2D, got shape: {X.shape}"
            )

        if X.shape[1] != 23:
            raise ValueError(
                f"FitAI VAE expects 23 features, "
                f"but dataset contains {X.shape[1]}"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "Dataset contains NaN or infinite values"
            )

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # Захист від ділення на нуль для постійних колонок
        self.std = np.where(
            self.std < 1e-8,
            1.0,
            self.std
        )

        X = (X - self.mean) / self.std

        return X

    # --------------------------------------------------------

    def train(self):

        X = self.load_dataset()

        history = []

        print(
            f"Training samples: {X.shape[0]} | "
            f"Features: {X.shape[1]} | "
            f"Epochs: {self.epochs} | "
            f"Learning rate: {self.learning_rate} | "
            f"Beta: {self.beta}"
        )

        for epoch in range(self.epochs):

            self.model.zero_grad()

            result = self.model.forward(X)

            reconstruction = result["reconstruction"]
            latent = result["latent_states"]

            losses = self.loss_fn.total_loss(
                X,
                reconstruction,
                latent
            )

            grads = self.loss_fn.backward(
                X,
                reconstruction,
                latent
            )

            self.model.backward(grads)

            self.model.update(
                self.learning_rate
            )

            total_loss = float(
                losses["total_loss"]
            )

            if not np.isfinite(total_loss):
                raise FloatingPointError(
                    f"Loss became NaN or infinity "
                    f"at epoch {epoch}"
                )

            history.append(total_loss)

            if epoch % 10 == 0 or epoch == self.epochs - 1:

                print(
                    f"Epoch {epoch:4d} | "
                    f"Total={losses['total_loss']:.6f} | "
                    f"Global={losses['global_loss']:.6f} | "
                    f"Energy={losses['energy_loss']:.6f} | "
                    f"Recovery={losses['recovery_loss']:.6f} | "
                    f"Stress={losses['stress_loss']:.6f} | "
                    f"Muscle={losses['muscle_loss']:.6f} | "
                    f"Metabolism={losses['metabolism_loss']:.6f} | "
                    f"Aging={losses['aging_loss']:.6f} | "
                    f"KL={losses['kl_loss']:.6f}"
                )

        return history

    # --------------------------------------------------------

    def save(self, filename):

        if self.mean is None or self.std is None:
            raise RuntimeError(
                "Model cannot be saved before training "
                "or dataset loading"
            )

        filename = Path(filename)

        filename.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        state = {
            "model": self.model.save(),
            "mean": self.mean,
            "std": self.std,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "beta": self.beta
        }

        with filename.open("wb") as f:
            pickle.dump(state, f)

        print(f"\nModel saved -> {filename}")


if __name__ == "__main__":

    trainer = VAETrainer(
        csv_path=(
            r"C:\FitAI_v2\data"
            r"\edited_23_params_realistic.csv"
        ),
        learning_rate=0.001,
        epochs=200,
        beta=1.0
    )

    history = trainer.train()

    trainer.save(
        r"C:\FitAI_v2\ml\models\fitai_vae.pkl"
    )