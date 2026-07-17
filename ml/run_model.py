"""
FitAI Research

run_model.py

Quick diagnostic test for the complete Hierarchical VAE.
"""

import numpy as np

from ml.vae.model import FitAIVAE


def main():

    print("=" * 60)
    print("FitAI Hierarchical VAE Diagnostic")
    print("=" * 60)

    # -------------------------------------------------
    # Create model
    # -------------------------------------------------

    model = FitAIVAE()

    print("✓ Model created")

    # -------------------------------------------------
    # Random physiological batch
    # -------------------------------------------------

    x = np.random.rand(2, 23).astype(np.float64)

    print(f"Input shape           : {x.shape}")

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------

    result = model.forward(x)

    print("\nForward pass completed.\n")

    # -------------------------------------------------
    # Local latent states
    # -------------------------------------------------

    latent = result["latent_states"]

    print("Local Latent States")

    print(f" Energy      : {latent['energy'].shape}")
    print(f" Recovery    : {latent['recovery'].shape}")
    print(f" Stress      : {latent['stress'].shape}")
    print(f" Muscle      : {latent['muscle'].shape}")
    print(f" Metabolism  : {latent['metabolism'].shape}")
    print(f" Aging       : {latent['aging'].shape}")

    # -------------------------------------------------
    # Bioenergetic Core
    # -------------------------------------------------

    bai = result["bai"]

    print("\nBioenergetic Core")

    print(f" Latent Vector : {bai['latent_vector'].shape}")
    print(f" μ             : {bai['mu'].shape}")
    print(f" log_var       : {bai['log_var'].shape}")
    print(f" BAI           : {bai['bai'].shape}")

    # -------------------------------------------------
    # Decoder
    # -------------------------------------------------

    reconstruction = result["reconstruction"]

    print("\nDecoder")

    print(f" Reconstruction : {reconstruction.shape}")

    # -------------------------------------------------
    # Finished
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("Hierarchical VAE works correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()