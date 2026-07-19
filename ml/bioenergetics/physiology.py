"""Deterministic physiological interpretation of FitAI's BAI latent vector.

The Bioenergetic Core defines BAI as an unlabelled four-dimensional Gaussian
latent vector.  Individual coordinates therefore do not have distinct
physiological meanings.  This module deliberately treats them symmetrically.
All public physiological scores use a 0--100 scale (higher means more of the
named quantity).
"""

from collections.abc import Mapping, Sequence

import numpy as np


LOCAL_STATE_NAMES = (
    "energy",
    "recovery",
    "stress",
    "muscle",
    "metabolism",
    "aging",
)
BAI_SHAPE = (4,)
# Values outside this interval are far beyond the useful numerical range of a
# standard Gaussian VAE latent and almost certainly indicate corrupt input.
BAI_MIN = -20.0
BAI_MAX = 20.0


class PhysiologyState:
    """A deterministic 0--100 interpretation of the existing global BAI.

    BAI coordinates are exchangeable and unlabelled in :class:`BioenergeticCore`.
    We consequently apply the same logistic transform to every coordinate and
    average them.  This preserves the model's global adaptation semantics
    without inventing coordinate-specific meanings.
    """

    def __init__(self, bai: Sequence[float]):
        raw = np.asarray(bai)
        if raw.shape != BAI_SHAPE:
            raise ValueError("bai must be a one-dimensional array of 4 values")
        if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
            raw.dtype, np.complexfloating
        ):
            raise TypeError("bai must have a real numeric dtype")

        values = np.array(raw, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(values)):
            raise ValueError("bai values must be finite")
        if np.any(values < BAI_MIN) or np.any(values > BAI_MAX):
            raise ValueError(
                f"bai values must be between {BAI_MIN:g} and {BAI_MAX:g}"
            )
        values.setflags(write=False)
        self._bai = values

        # Stable logistic for the documented, bounded input range.
        component_scores = np.empty_like(values)
        positive = values >= 0
        component_scores[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exp_values = np.exp(values[~positive])
        component_scores[~positive] = exp_values / (1.0 + exp_values)
        self._adaptation = float(100.0 * np.mean(component_scores))

    @property
    def bai(self) -> np.ndarray:
        """Return a copy so callers cannot mutate the validated state."""
        return self._bai.copy()

    @property
    def adaptation(self) -> float:
        return self._adaptation

    @property
    def recovery(self) -> float:
        return self._adaptation

    @property
    def stress(self) -> float:
        return 100.0 - self._adaptation

    @property
    def fatigue(self) -> float:
        return 100.0 - self._adaptation

    @property
    def performance(self) -> float:
        return self._adaptation

    @property
    def readiness(self) -> float:
        return self.performance

    def to_dict(self) -> dict[str, float]:
        """Return all physiological scores on the documented 0--100 scale."""
        return {
            "fatigue": self.fatigue,
            "recovery": self.recovery,
            "stress": self.stress,
            "performance": self.performance,
            "readiness": self.readiness,
            "adaptation": self.adaptation,
        }


def build_physiological_state(
    features: Mapping[str, float],
    local_latents: Mapping[str, Sequence[float]],
    bai: Sequence[float],
    stress_trajectory: Sequence[float],
) -> dict:
    """Build the validated state consumed by the HJB decision-support stage."""
    trajectory = np.asarray(stress_trajectory, dtype=np.float64)
    if trajectory.shape != (7,):
        raise ValueError("stress_trajectory must contain exactly 7 daily values")
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("Physiological state values must be finite")

    local_state_means = {}
    for name in LOCAL_STATE_NAMES:
        values = np.asarray(local_latents[name], dtype=np.float64)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} latent state must contain 4 finite values")
        local_state_means[name] = float(np.mean(values))

    interpreted = PhysiologyState(bai)
    scores = interpreted.to_dict()
    return {
        "hrv": float(features["HRV"]),
        "sleep_hours": float(features["Sleep_hours"]),
        "systolic_bp": float(features["Systolic_blood_pressure_mmhg"]),
        "crp_mg_l": float(features["CRP_mg_l"]),
        "daily_stress": trajectory.tolist(),
        "local_state_means": local_state_means,
        "bai": interpreted.bai.tolist(),
        "scores": scores,
        **scores,
    }
