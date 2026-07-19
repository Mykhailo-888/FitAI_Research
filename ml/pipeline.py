"""End-to-end integration pipeline for FitAI's legacy and latent models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ml.bioenergetics.latent_states import FEATURE_INDEX
from ml.bioenergetics.physiology import build_physiological_state
from ml.emotional_drift import analyze_emotional_drift
from ml.hjb.hjb_optimizer import (
    CRITICAL_CRP_MG_L,
    LOW_HRV,
    optimize_hjb_trajectory,
)
from ml.photo_analysis import analyze_body_proportions
from ml.training_optimizer import weekly_training_plan_optimizer
from ml.vae.model import FitAIVAE


CANONICAL_FEATURES = tuple(
    name for name, _ in sorted(FEATURE_INDEX.items(), key=lambda item: item[1])
)
LOCAL_STATE_NAMES = (
    "energy",
    "recovery",
    "stress",
    "muscle",
    "metabolism",
    "aging",
)
FEATURE_RANGES = {
    "Age": (16, 100),
    "Height_cm": (100, 250),
    "Weight_kg": (30, 300),
    "Waist_circumference_cm": (40, 200),
    "Emotional_stress": (1, 10),
    "Alcohol_units_per_week": (0, 100),
    "Daily_calories_kcal": (0, 20000),
    "Max_push_ups": (0, 500),
    "Max_pull_ups": (0, 100),
    "Run_1km_min": (3, 30),
    "Run_100m_sec": (10, 60),
    "Cooper_test_km": (1, 20),
    "Burpees_3min": (0, 200),
    "Push_ups_1min": (0, 100),
    "Sleep_hours": (0, 24),
    "Resting_heart_rate_bpm": (30, 120),
    "Systolic_blood_pressure_mmhg": (80, 200),
    "Mitochondria_placeholder": (0, 1000),
    "Testosterone_ng_dl": (0, 2000),
    "Cortisol_ug_dl": (0, 100),
    "Hemoglobin_g_dl": (0, 25),
    "CRP_mg_l": (0, 100),
    "HRV": (0, 200),
}


@contextmanager
def _numpy_seed(seed: int | None):
    """Temporarily seed legacy NumPy-based VAE layers without leaking state."""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _canonical_dict(features: Mapping[str, Any] | Sequence[float]) -> dict[str, float]:
    if isinstance(features, Mapping):
        missing = [name for name in CANONICAL_FEATURES if name not in features]
        if missing:
            raise ValueError(f"Missing canonical features: {missing}")
        values = []
        for name in CANONICAL_FEATURES:
            value = features[name]
            if value is None or value == "":
                raise ValueError(f"Missing value for canonical feature: {name}")
            try:
                values.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
        canonical = dict(zip(CANONICAL_FEATURES, values))
    else:
        try:
            values = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("All canonical features must be numeric") from exc
        if values.ndim == 2 and values.shape == (1, 23):
            values = values[0]
        if values.ndim != 1 or values.shape[0] != 23:
            raise ValueError("Expected a mapping or a sequence of exactly 23 features")
        canonical = dict(
            zip(CANONICAL_FEATURES, (float(value) for value in values))
        )

    for name, value in canonical.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        minimum, maximum = FEATURE_RANGES[name]
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{name} must be between {minimum} and {maximum}"
            )
    return canonical


def _photo_result(photo_path: str | Path | None, warnings: list[str]) -> dict:
    if photo_path is None:
        return {
            "provided": False,
            "auxiliary_body_proportions": {},
            "error": None,
        }
    try:
        result = analyze_body_proportions(str(photo_path))
    except Exception as exc:  # OpenCV/filesystem failures must not stop inference.
        result = {"error": f"Photo analysis failed: {exc}"}

    if result.get("error"):
        warnings.append(str(result["error"]))
    auxiliary = {
        key: result[key]
        for key in (
            "shoulder_width_px",
            "waist_width_px",
            "hip_width_px",
            "shoulder_waist_ratio",
            "waist_hip_ratio",
            "body_type",
            "recommendation",
        )
        if key in result
    }
    return {
        "provided": True,
        "auxiliary_body_proportions": auxiliary,
        **result,
    }


def _bounded_vae_input(features: Mapping[str, float]) -> np.ndarray:
    """Map heterogeneous raw units to the finite range expected by the VAE."""
    raw = np.asarray(
        [[features[name] for name in CANONICAL_FEATURES]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(raw)):
        raise ValueError("All canonical features must be finite numeric values")
    return raw / (1.0 + np.abs(raw))


def run_pipeline(
    features: Mapping[str, Any] | Sequence[float],
    photo_path: str | Path | None = None,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    vae_model: FitAIVAE | None = None,
) -> dict:
    """Run the complete FitAI pipeline without mutating caller-owned inputs."""
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both")

    original = deepcopy(features)
    input_features = _canonical_dict(original)
    updated_features = dict(input_features)
    warnings: list[str] = []

    photo_analysis = _photo_result(photo_path, warnings)
    drift = analyze_emotional_drift(
        input_features["Emotional_stress"],
        input_features["Alcohol_units_per_week"],
        days=7,
        rng=rng,
        seed=seed,
        return_details=True,
    )
    updated_features["Emotional_stress"] = float(drift["mean_stress"])

    x = _bounded_vae_input(updated_features)
    if x.shape != (1, 23):
        raise RuntimeError(f"Canonical VAE input has invalid shape {x.shape}")

    with _numpy_seed(seed):
        model = vae_model if vae_model is not None else FitAIVAE()
        vae_result = model.forward(x)

    latent_arrays = {
        name: np.asarray(vae_result["latent_states"][name], dtype=float)
        for name in LOCAL_STATE_NAMES
    }
    if any(value.shape != (1, 4) for value in latent_arrays.values()):
        shapes = {name: value.shape for name, value in latent_arrays.items()}
        raise RuntimeError(f"Invalid local latent shapes: {shapes}")
    local_latents = {
        name: value[0].tolist() for name, value in latent_arrays.items()
    }
    bai_array = np.asarray(vae_result["bai"]["bai"], dtype=float)
    latent_vector = np.asarray(vae_result["bai"]["latent_vector"], dtype=float)
    if bai_array.shape != (1, 4) or latent_vector.shape != (1, 24):
        raise RuntimeError(
            f"Invalid Bioenergetic Core shapes: BAI={bai_array.shape}, "
            f"local={latent_vector.shape}"
        )
    bai = bai_array[0].tolist()

    physiological_state = build_physiological_state(
        updated_features, local_latents, bai, drift["trajectory"]
    )
    hjb = optimize_hjb_trajectory(physiological_state)
    warnings.extend(hjb["warnings"])
    weekly_plan = weekly_training_plan_optimizer(
        current_weight=updated_features["Weight_kg"],
        target_weight=updated_features["Weight_kg"],
        current_hrv=updated_features["HRV"],
        sleep_hours=updated_features["Sleep_hours"],
        alcohol_units=updated_features["Alcohol_units_per_week"],
        age=updated_features["Age"],
        systolic_bp=updated_features["Systolic_blood_pressure_mmhg"],
        resting_bpm=updated_features["Resting_heart_rate_bpm"],
    )
    weekly_plan["optimal_intensity"] = hjb["optimal_intensity"]
    weekly_plan["training_risk"] = hjb["risk"]

    if hjb["critical_crp_override"]:
        weekly_plan.update(
            {
                "weekly_hiit_sessions": 0,
                "other_sessions": 0,
                "optimal_intensity": 0.0,
                "hiit_recommendation": "No training; seek medical guidance",
                "training_recommendation": "medical_review",
            }
        )
    elif updated_features["HRV"] < LOW_HRV:
        weekly_plan["weekly_hiit_sessions"] = 0
        weekly_plan["hiit_recommendation"] = "Avoid HIIT until HRV recovers"

    return {
        "input_features": input_features,
        "updated_features": updated_features,
        "photo_analysis": photo_analysis,
        "emotional_drift": drift,
        "local_latents": local_latents,
        "bai": bai,
        "physiological_state": physiological_state,
        "hjb": hjb,
        "weekly_plan": weekly_plan,
        "warnings": warnings,
    }


class FitAIPipeline:
    """Reusable wrapper around :func:`run_pipeline`."""

    def __init__(self, vae_model: FitAIVAE | None = None):
        self.vae_model = vae_model

    def run(self, features, photo_path=None, *, seed=None, rng=None):
        return run_pipeline(
            features,
            photo_path,
            seed=seed,
            rng=rng,
            vae_model=self.vae_model,
        )
