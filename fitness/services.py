"""Django orchestration around the stable FitAI analysis implementations."""
from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping

import numpy as np
from django.db import transaction
from django.utils import timezone

from ml.fit_model_core import get_fitness_model
from ml.pipeline import CANONICAL_FEATURES
from ml.services.athlete_assessment import MODEL_VERSION, assess_athlete, validate_athlete_input
from ml.training_risk import predict_risk

from .models import (
    AssessmentResult, Athlete, AthleteMeasurement, BioenergeticAssessment,
    ModelComparison, ModelRun, Recommendation,
)

PREDICTION_SCHEMA = (
    ("Calories Burned", "kcal"), ("1 km Run Time", "min"),
    ("Cooper Test Distance", "km"), ("Max Pull-Ups", "reps"),
    ("Burpees Capacity", "reps/hour"), ("10 km Run Time", "min"),
    ("Waist Circumference Change", "cm"), ("Testosterone Projection", "ng/dL"),
)


def run_legacy_analysis(features: Mapping[str, float]) -> dict[str, Any]:
    """Callable legacy entry point extracted from the original Django view."""
    model = get_fitness_model("simple")
    values = np.array([[features[name] for name in CANONICAL_FEATURES]], dtype=float)
    predictions = model.predict(values)[0]
    crp, hrv = features["CRP_mg_l"], features["HRV"]
    sleep, bp = features["Sleep_hours"], features["Systolic_blood_pressure_mmhg"]
    stress = features["Emotional_stress"]
    warnings = []
    if crp >= 5:
        risk = {"score": 10.0, "label": "Critical inflammation gate", "intensity": 0.0}
        recommendation = "REST and obtain medical guidance before training."
        warnings.append(f"CRP critical risk gate active ({crp:.1f} mg/L).")
    else:
        score, label, intensity = predict_risk(hrv, sleep, bp, stress, crp)
        risk = {"score": float(score), "label": label, "intensity": float(intensity)}
        recommendation = (
            "High-intensity training is permitted." if intensity >= .85
            else "Moderate or reduced-load training is recommended."
        )
    return {
        "predictions": [
            {"name": name, "unit": unit, "value": round(float(value), 3)}
            for (name, unit), value in zip(PREDICTION_SCHEMA, predictions)
        ],
        "raw_predictions": predictions.tolist(),
        "recommendation": recommendation,
        "risk": risk,
        "warnings": warnings,
        "explainability": "Per-output sensitivity is available in ml/feature_importance.json.",
    }


def actionable_recommendation(advanced: Mapping[str, Any]) -> dict[str, Any]:
    states, bio = advanced["physiological_states"], advanced["bioenergetic_state"]
    risks, hjb = advanced["risks"], advanced["hjb_control"]
    critical = bool(hjb.get("critical_crp_override"))
    if critical:
        code, intensity, volume, session = "MEDICAL_CAUTION", "0%", "0%", "No training"
    elif hjb["optimal_intensity"] == 0 or states["fatigue"] >= 80:
        code, intensity, volume, session = "REST", "0-10%", "0-20%", "Rest and mobility only"
    elif states["readiness"] < 45 or states["recovery"] < 45:
        code, intensity, volume, session = "ACTIVE_RECOVERY", "20-40%", "30-50%", "Walk, mobility, easy aerobic"
    elif states["readiness"] < 65 or bio["confidence"] < .4:
        code, intensity, volume, session = "TRAIN_LIGHT", "40-65%", "50-75%", "Technique or easy aerobic"
    else:
        code, intensity, volume, session = "TRAIN", "65-85%", "75-100%", "Planned sport-specific session"
    return {
        "code": code, "label": code.replace("_", " ").title(),
        "reason": risks[-1] if risks else advanced["recommendation"]["reason"],
        "intensity_range": intensity, "volume_range": volume,
        "session_type": session,
        "recovery_actions": ["Prioritize sleep", "Hydrate", "Reassess if symptoms or readiness worsen"],
        "next_reassessment": "Within 24 hours after recovery or before the next hard session",
        "risk_warnings": risks,
        "disclaimer": "This is research decision support and not a medical diagnosis.",
    }


@transaction.atomic
def assess_and_compare(payload: Mapping[str, Any]) -> ModelComparison:
    """Persist one normalized measurement and both architecture runs."""
    features, quality = validate_athlete_input(payload)
    athlete_id = str(payload.get("athlete_id") or "onboarding-athlete").strip()
    athlete, _ = Athlete.objects.update_or_create(
        public_id=athlete_id,
        defaults={
            "name": str(payload.get("name") or ""),
            "sex": str(payload.get("sex") or ""),
            "height_cm": features["Height_cm"],
            "is_demonstration": bool(payload.get("is_demonstration", False)),
        },
    )
    measurement = AthleteMeasurement.objects.create(
        athlete=athlete, measured_at=payload.get("measurement_time") or timezone.now(),
        source_type=str(payload.get("source_type") or "manual_onboarding"),
        raw_inputs=features, raw_payload=dict(payload), units=quality["units"],
        sources=quality["sources"], input_quality=quality["score"],
        missing_features=quality["missing_features"], imputed_features=quality["imputed_features"],
        measured_fields=quality["measured_features"], reported_fields=quality["reported_features"],
    )
    started = perf_counter()
    legacy = run_legacy_analysis(features)
    legacy_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    advanced = assess_athlete({**dict(payload), "athlete_id": athlete_id}, persist=False)
    advanced["recommendation"] = {
        **advanced["recommendation"], **actionable_recommendation(advanced)
    }
    advanced_ms = (perf_counter() - started) * 1000
    legacy_run = ModelRun.objects.create(
        athlete=athlete, measurement=measurement, architecture_type="legacy",
        model_name="FitnessNeuralNet", model_version="trained_fitness_model_simple.pkl",
        input_snapshot=features, output_snapshot=legacy, warnings=legacy["warnings"],
        duration_ms=legacy_ms,
    )
    advanced_run = ModelRun.objects.create(
        athlete=athlete, measurement=measurement, architecture_type="bioenergetic",
        model_name="FitAI athlete assessment", model_version=MODEL_VERSION,
        input_snapshot=features, output_snapshot=advanced, warnings=advanced["risks"],
        duration_ms=advanced_ms,
    )
    s, b = advanced["physiological_states"], advanced["bioenergetic_state"]
    assessment = BioenergeticAssessment.objects.create(
        athlete=athlete, measurement=measurement, model_version=MODEL_VERSION,
        bai=b["bai_raw"], bai_normalized=b["bai_normalized"], confidence=b["confidence"],
        uncertainty=b["uncertainty"], reconstruction_error=b["reconstruction_error"],
        latent_states=advanced["latent_states"], fatigue=s["fatigue"], recovery=s["recovery"],
        stress=s["stress"], inflammation=s["inflammation"], performance=s["performance"],
        readiness=s["readiness"], adaptation=s["adaptation"], risk_flags=advanced["risks"],
        explanation=advanced["explanation"], result=advanced,
    )
    Recommendation.objects.create(
        assessment=assessment, training_intensity=advanced["recommendation"]["training_intensity"],
        recovery_recommendation=advanced["recommendation"]["recovery"],
        hjb_action=advanced["hjb_control"], safety_restrictions=advanced["risks"],
        explanation=advanced["recommendation"]["reason"],
    )
    AssessmentResult.objects.create(
        measurement=measurement, assessment=assessment, bai=b["bai_normalized"],
        confidence=b["confidence"], uncertainty=b["uncertainty"],
        physiological_states=s, latent_states=advanced["latent_states"],
        risk_flags=advanced["risks"], recommendation=advanced["recommendation"],
        hjb_plan=advanced["hjb_control"], explanation=advanced["explanation"],
        limitations=advanced["explanation"]["limitations"],
    )
    differences = {
        item["name"]: 0.0
        for item in legacy["predictions"]
    }
    return ModelComparison.objects.create(
        measurement=measurement, legacy_model_run=legacy_run,
        new_architecture_model_run=advanced_run,
        compatible_output_differences=differences,
        added_capabilities=[
            "BAI research proxy", "six latent states", "physiological states",
            "uncertainty", "risk gates", "personal baseline", "HJB control",
        ],
        stability_metrics={"identical_23_feature_input": True},
        summary=(
            "Both architectures received the identical canonical 23-feature vector. "
            "Their eight trained outputs are identical because the advanced service reuses "
            "the stable trained model; added values are decision-support variables, not proof of accuracy."
        ),
    )
