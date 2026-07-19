# BAI interpretation

BAI is a research proxy for latent bioenergetic adaptation. It is not a directly
measured mitochondrial quantity, medical diagnosis, or clinically validated score.

`bai_raw` comes from the existing four-dimensional Bioenergetic Core. The coordinates
have no trained coordinate-specific labels and are treated symmetrically.
`bai_normalized` is the existing 0-100 presentation transform, not an empirical
clinical percentile. Always show it with confidence, input quality and limitations.

No serialized trained hierarchical VAE weights exist in this repository. The latent
results are reproducible proof-of-concept signals. The service uses the saved trained
23-to-8 model in parallel, penalizes confidence, and identifies the model as
`trained-fitness-simple+poc-hierarchical-vae-untrained`.

Personal trends require three prior assessments. Population comparison is partial
and does not calibrate BAI. A validated release requires a versioned hierarchy
checkpoint and a provenance-verified calibration cohort.
