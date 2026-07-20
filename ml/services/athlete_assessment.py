"""Canonical athlete validation, assessment, reference and persistence service."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ml.bioenergetics.latent_states import FEATURE_INDEX
from ml.fit_model_core import get_fitness_model
from ml.pipeline import CANONICAL_FEATURES, FEATURE_RANGES, FitAIPipeline
from ml.vae.model import FitAIVAE

ROOT = Path(__file__).resolve().parents[2]
MODEL_VERSION = "trained-fitness-simple+poc-hierarchical-vae-untrained"
DISCLAIMER = "Research estimate, not a medical diagnosis."

UNITS = {
    "Age": "years", "Height_cm": "cm", "Weight_kg": "kg",
    "Waist_circumference_cm": "cm", "Emotional_stress": "1-10",
    "Alcohol_units_per_week": "units/week", "Daily_calories_kcal": "kcal/day",
    "Max_push_ups": "repetitions", "Max_pull_ups": "repetitions",
    "Run_1km_min": "minutes", "Run_100m_sec": "seconds",
    "Cooper_test_km": "km", "Burpees_3min": "repetitions",
    "Push_ups_1min": "repetitions", "Sleep_hours": "hours/day",
    "Resting_heart_rate_bpm": "bpm", "Systolic_blood_pressure_mmhg": "mmHg",
    "Mitochondria_placeholder": "legacy_unitless_proxy",
    "Testosterone_ng_dl": "ng/dL", "Cortisol_ug_dl": "ug/dL",
    "Hemoglobin_g_dl": "g/dL", "CRP_mg_l": "mg/L", "HRV": "ms",
}
ALIASES = {
    "age": "Age", "height_cm": "Height_cm", "weight_kg": "Weight_kg",
    "waist_cm": "Waist_circumference_cm", "waist_circumference_cm": "Waist_circumference_cm",
    "perceived_stress": "Emotional_stress", "emotional_stress": "Emotional_stress",
    "alcohol_consumption": "Alcohol_units_per_week", "alcohol_units_per_week": "Alcohol_units_per_week",
    "daily_calories_kcal": "Daily_calories_kcal", "max_push_ups": "Max_push_ups",
    "push_ups": "Max_push_ups", "max_pull_ups": "Max_pull_ups", "pull_ups": "Max_pull_ups",
    "run_1km_min": "Run_1km_min", "run_100m_sec": "Run_100m_sec",
    "cooper_test_km": "Cooper_test_km", "burpees_3min": "Burpees_3min",
    "push_ups_1min": "Push_ups_1min", "sleep_hours": "Sleep_hours",
    "resting_heart_rate_bpm": "Resting_heart_rate_bpm",
    "systolic_blood_pressure_mmhg": "Systolic_blood_pressure_mmhg",
    "mitochondria_placeholder": "Mitochondria_placeholder",
    "testosterone_ng_dl": "Testosterone_ng_dl", "cortisol_ug_dl": "Cortisol_ug_dl",
    "hemoglobin_g_dl": "Hemoglobin_g_dl", "crp_mg_l": "CRP_mg_l", "hrv": "HRV",
}


@lru_cache(maxsize=1)
def _imputation_medians() -> dict[str, float]:
    frame = pd.read_csv(ROOT / "data" / "edited_23_params_realistic.csv")
    return {name: float(frame[name].median()) for name in CANONICAL_FEATURES}


@lru_cache(maxsize=1)
def _models():
    # The hierarchical artifact is absent. A fixed seed makes the repository's
    # proof-of-concept architecture reproducible without pretending it is trained.
    state = np.random.get_state()
    np.random.seed(20260719)
    vae = FitAIVAE()
    np.random.set_state(state)
    return FitAIPipeline(vae), get_fitness_model("simple")


def validate_athlete_input(payload: Mapping[str, Any]) -> tuple[dict, dict]:
    supplied = {}
    sources_in = payload.get("sources", {})
    units_in = payload.get("units", {})
    warnings, missing, imputed = [], [], {}
    measured, reported = [], []
    for key, value in payload.items():
        canonical = ALIASES.get(key, key if key in FEATURE_INDEX else None)
        if canonical and value not in (None, ""):
            supplied[canonical] = value

    medians = _imputation_medians()
    canonical = {}
    source_map = {}
    for name in CANONICAL_FEATURES:
        if name not in supplied:
            missing.append(name)
            canonical[name] = medians[name]
            imputed[name] = {"value": medians[name], "source": "dataset-imputed"}
            source_map[name] = "dataset-imputed"
            continue
        try:
            value = float(supplied[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        low, high = FEATURE_RANGES[name]
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high} {UNITS[name]}")
        supplied_unit = units_in.get(name) or units_in.get(next((k for k, v in ALIASES.items() if v == name), ""), UNITS[name])
        if supplied_unit != UNITS[name]:
            raise ValueError(f"{name} requires unit {UNITS[name]}, received {supplied_unit}")
        canonical[name] = value
        source = sources_in.get(name, "athlete-reported")
        if source not in {"measured", "athlete-reported", "image-estimated", "dataset-imputed"}:
            raise ValueError(f"Unsupported source for {name}: {source}")
        source_map[name] = source
        (measured if source == "measured" else reported).append(name)

    if "Mitochondria_placeholder" in imputed:
        warnings.append("Legacy mitochondria placeholder was imputed; it is not a measurement.")
    warnings.append("Hierarchical VAE weights are not present; latent states are proof-of-concept only.")
    completeness = (len(CANONICAL_FEATURES) - len(missing)) / len(CANONICAL_FEATURES)
    measured_fraction = len(measured) / len(CANONICAL_FEATURES)
    score = round(100 * (0.75 * completeness + 0.25 * measured_fraction), 1)
    quality = {
        "score": score, "measured_features": measured, "reported_features": reported,
        "imputed_features": imputed, "missing_features": missing, "warnings": warnings,
        "sources": source_map, "units": UNITS,
    }
    return canonical, quality


@lru_cache(maxsize=1)
def _reference_frame() -> pd.DataFrame:
    path = ROOT / "data" / "gym_members_exercise_tracking.csv"
    return pd.read_csv(path)


def _reference(features: Mapping[str, float]) -> dict:
    frame = _reference_frame()
    mappings = {
        "Age": "Age", "Weight_kg": "Weight (kg)",
        "Height_cm": "Height (m)", "Resting_heart_rate_bpm": "Resting_BPM",
    }
    percentiles, z_scores = {}, {}
    for feature, column in mappings.items():
        values = frame[column].astype(float)
        target = features[feature] / 100 if feature == "Height_cm" else features[feature]
        median = float(values.median())
        mad = float(np.median(np.abs(values - median)))
        percentiles[feature] = round(float(100 * (values <= target).mean()), 1)
        z_scores[feature] = round(float(0.6745 * (target - median) / mad), 3) if mad else None
    return {
        "status": "partial_reference_data", "cohort": "gym_members_exercise_tracking",
        "cohort_size": int(len(frame)), "percentiles": percentiles,
        "robust_z_scores": z_scores,
        "limitations": ["Exact upstream URL and license are absent from the repository.",
                        "Only age, height, weight and resting heart rate are comparable."],
    }


def _personal_baseline(athlete_id: str, current: dict) -> dict:
    try:
        from fitness.models import Athlete
        athlete = Athlete.objects.filter(public_id=athlete_id).first()
        rows = list(athlete.assessments.order_by("-created_at")[:28]) if athlete else []
    except Exception:
        rows = []
    if len(rows) < 3:
        return {"status": "insufficient_history", "history_count": len(rows),
                "minimum_required": 3, "limitations": ["No reliable personal baseline yet."]}
    keys = ("bai_normalized", "readiness", "recovery", "stress")
    medians = {key: float(np.median([getattr(row, key) for row in rows])) for key in keys}
    return {"status": "available", "history_count": len(rows), "rolling_median": medians,
            "deviation": {key: round(float(current[key] - value), 2) for key, value in medians.items()}}


def assess_athlete(payload: Mapping[str, Any], *, persist: bool = False, seed: int = 20260719) -> dict:
    athlete_id = str(payload.get("athlete_id") or "").strip()
    if not athlete_id:
        raise ValueError("athlete_id is required")
    features, quality = validate_athlete_input(payload)
    pipeline, trained_model = _models()
    result = pipeline.run(features, seed=seed)
    trained_prediction = trained_model.predict(np.array([[features[n] for n in CANONICAL_FEATURES]]))[0]
    reconstruction = np.asarray(result["physiological_state"]["bai"], dtype=float)
    bai_raw = float(np.mean(reconstruction))
    # Existing PhysiologyState supplies the repository's documented presentation transform.
    bai_normalized = float(result["physiological_state"]["adaptation"])
    vae_x = np.array([features[n] for n in CANONICAL_FEATURES], dtype=float)
    vae_x = vae_x / (1 + np.abs(vae_x))
    with np.errstate(all="ignore"):
        recon = np.asarray(result.get("reconstruction", np.zeros(23)), dtype=float)
    # run_pipeline currently exposes local reconstructions but not the global one.
    reconstruction_error = float(np.mean(np.square(vae_x))) if recon.size != 23 else float(np.mean((vae_x - recon) ** 2))

    stress_input = features["Emotional_stress"] * 10
    recovery = np.clip(0.45 * result["physiological_state"]["recovery"] + 4 * features["Sleep_hours"] + 0.25 * features["HRV"] - 15, 0, 100)
    inflammation = np.clip(features["CRP_mg_l"] * 12, 0, 100)
    stress = np.clip(0.55 * stress_input + 0.25 * (100 - recovery) + 0.2 * min(features["Cortisol_ug_dl"] * 4, 100), 0, 100)
    fatigue = np.clip(0.55 * (100 - recovery) + 0.3 * stress + 0.15 * inflammation, 0, 100)
    performance = np.clip(0.45 * bai_normalized + 0.35 * recovery + 0.2 * (100 - fatigue), 0, 100)
    readiness = np.clip(0.4 * recovery + 0.3 * performance + 0.15 * (100 - stress) + 0.15 * (100 - inflammation), 0, 100)
    states = {k: round(float(v), 2) for k, v in {
        "fatigue": fatigue, "recovery": recovery, "stress": stress,
        "inflammation": inflammation, "performance": performance,
        "readiness": readiness, "adaptation": bai_normalized}.items()}
    risks = list(result["warnings"])
    if features["Systolic_blood_pressure_mmhg"] >= 160:
        risks.append("High systolic blood-pressure safety gate: no intense training.")
    if states["fatigue"] >= 80 or states["stress"] >= 80:
        risks.append("Severe fatigue/stress gate: recovery session only.")
    confidence = np.clip((quality["score"] / 100) * 0.65 + 0.1, 0.05, 0.75)
    confidence *= 0.75  # absent trained hierarchical artifact
    reference = _reference(features)
    if reference["status"] != "matched_reference_data":
        confidence *= 0.9
    confidence = round(float(confidence), 3)
    hjb = result["hjb"]
    intensity = float(hjb["optimal_intensity"])
    if features["Systolic_blood_pressure_mmhg"] >= 160 or states["fatigue"] >= 80:
        intensity = 0.0
    recommendation = {
        "training_intensity": round(intensity, 3),
        "training_volume": "rest" if intensity == 0 else ("reduced" if intensity < .5 else "maintain"),
        "recovery": "Prioritize sleep, hydration, and reassessment before progression.",
        "action": "rest_and_reassess" if intensity == 0 else "controlled_training",
        "reason": risks[-1] if risks else "No critical gate; intensity is limited by recovery/readiness.",
        "clinically_optimal": False,
    }
    positive = [n for n, good in [("sleep", features["Sleep_hours"] >= 7), ("HRV", features["HRV"] >= 60), ("CRP", features["CRP_mg_l"] < 3)] if good]
    negative = [n for n, bad in [("sleep deficit", features["Sleep_hours"] < 7), ("low HRV", features["HRV"] < 50), ("elevated CRP", features["CRP_mg_l"] >= 3), ("high stress", features["Emotional_stress"] >= 8)] if bad]
    output = {
        "athlete_id": athlete_id, "assessment_time": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION, "input_quality": quality,
        "bioenergetic_state": {"bai_raw": round(bai_raw, 6), "bai_normalized": round(bai_normalized, 2),
            "confidence": confidence, "uncertainty": round(1 - confidence, 3),
            "reconstruction_error": round(reconstruction_error, 6),
            "interpretation": "Experimental latent bioenergetic adaptation proxy; not a direct mitochondrial measurement."},
        "latent_states": {k: round(float(np.mean(v)), 6) for k, v in result["local_latents"].items()},
        "latent_vector": [x for values in result["local_latents"].values() for x in values],
        "physiological_states": states, "athlete_reference": reference,
        "personal_baseline": {}, "risks": risks, "hjb_control": {**hjb, "optimal_intensity": intensity},
        "recommendation": recommendation,
        "trained_model_outputs": trained_prediction.tolist(),
        "explanation": {"positive_factors": positive, "negative_factors": negative,
            "dominant_features": ["Sleep_hours", "HRV", "Emotional_stress", "CRP_mg_l"],
            "limitations": quality["warnings"] + reference["limitations"]},
        "disclaimer": DISCLAIMER,
    }
    output["personal_baseline"] = _personal_baseline(athlete_id, {
        "bai_normalized": bai_normalized, **states
    })
    if persist:
        _persist(payload, features, output)
    return output


def _persist(payload, features, output):
    from django.db import transaction
    from django.utils.dateparse import parse_datetime
    from fitness.models import Athlete, AthleteMeasurement, BioenergeticAssessment, Recommendation
    with transaction.atomic():
        athlete, _ = Athlete.objects.update_or_create(
            public_id=output["athlete_id"],
            defaults={"sex": payload.get("sex", ""), "sport_type": payload.get("sport_type", ""),
                      "training_level": payload.get("training_level", ""),
                      "training_years": payload.get("training_years"),
                      "is_demonstration": bool(payload.get("is_demonstration", False))},
        )
        measured_at = parse_datetime(str(payload.get("measurement_time", ""))) or datetime.now(timezone.utc)
        measurement = AthleteMeasurement.objects.create(
            athlete=athlete, measured_at=measured_at, raw_inputs=features,
            units=output["input_quality"]["units"], sources=output["input_quality"]["sources"],
            input_quality=output["input_quality"]["score"],
            missing_features=output["input_quality"]["missing_features"],
            imputed_features=output["input_quality"]["imputed_features"])
        s, b = output["physiological_states"], output["bioenergetic_state"]
        assessment = BioenergeticAssessment.objects.create(
            athlete=athlete, measurement=measurement, model_version=output["model_version"],
            bai=b["bai_raw"], bai_normalized=b["bai_normalized"], confidence=b["confidence"],
            uncertainty=b["uncertainty"], reconstruction_error=b["reconstruction_error"],
            latent_states=output["latent_states"], fatigue=s["fatigue"], recovery=s["recovery"],
            stress=s["stress"], inflammation=s["inflammation"], performance=s["performance"],
            readiness=s["readiness"], adaptation=s["adaptation"], risk_flags=output["risks"],
            explanation=output["explanation"], result=output)
        r = output["recommendation"]
        Recommendation.objects.create(
            assessment=assessment, training_intensity=r["training_intensity"],
            recovery_recommendation=r["recovery"], hjb_action=output["hjb_control"],
            safety_restrictions=output["risks"], explanation=r["reason"])
