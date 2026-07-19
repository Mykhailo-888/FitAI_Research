from copy import deepcopy
from unittest.mock import patch

import numpy as np

from ml.bioenergetics.latent_states import FEATURE_INDEX
from ml.emotional_drift import analyze_emotional_drift
from ml.pipeline import CANONICAL_FEATURES, run_pipeline


def _features(**overrides):
    values = {
        "Age": 30,
        "Height_cm": 175,
        "Weight_kg": 75,
        "Waist_circumference_cm": 90,
        "Emotional_stress": 5,
        "Alcohol_units_per_week": 0,
        "Daily_calories_kcal": 2500,
        "Max_push_ups": 20,
        "Max_pull_ups": 8,
        "Run_1km_min": 6.0,
        "Run_100m_sec": 15.0,
        "Cooper_test_km": 2.8,
        "Burpees_3min": 40,
        "Push_ups_1min": 30,
        "Sleep_hours": 7.0,
        "Resting_heart_rate_bpm": 70,
        "Systolic_blood_pressure_mmhg": 120,
        "Mitochondria_placeholder": 50,
        "Testosterone_ng_dl": 500,
        "Cortisol_ug_dl": 15.0,
        "Hemoglobin_g_dl": 14.5,
        "CRP_mg_l": 1.5,
        "HRV": 70,
    }
    values.update(overrides)
    return values


def test_pipeline_preserves_order_input_and_waist_without_photo():
    features = _features()
    before = deepcopy(features)
    result = run_pipeline(features, seed=11)

    assert features == before
    assert tuple(result["input_features"]) == CANONICAL_FEATURES
    assert CANONICAL_FEATURES == tuple(
        name for name, _ in sorted(FEATURE_INDEX.items(), key=lambda item: item[1])
    )
    assert result["updated_features"]["Waist_circumference_cm"] == 90
    assert result["photo_analysis"]["provided"] is False


def test_bad_photo_is_non_fatal_and_does_not_replace_waist():
    result = run_pipeline(_features(), photo_path="missing-photo.jpg", seed=3)

    assert result["photo_analysis"]["error"] == "Photo not found"
    assert result["updated_features"]["Waist_circumference_cm"] == 90
    assert result["warnings"]


def test_mocked_photo_analysis_is_auxiliary_only():
    analysis = {
        "shoulder_width_px": 120,
        "waist_width_px": 80,
        "hip_width_px": 100,
        "shoulder_waist_ratio": 1.5,
        "waist_hip_ratio": 0.8,
        "body_type": "V-shape",
        "recommendation": "Decision-support metadata only.",
        "error": None,
    }
    with patch("ml.pipeline.analyze_body_proportions", return_value=analysis):
        result = run_pipeline(_features(), photo_path="mock.jpg", seed=3)

    assert result["photo_analysis"]["waist_width_px"] == 80
    assert result["updated_features"]["Waist_circumference_cm"] == 90


def test_photo_analyzer_exception_is_non_fatal():
    with patch(
        "ml.pipeline.analyze_body_proportions",
        side_effect=RuntimeError("OpenCV failed"),
    ):
        result = run_pipeline(_features(), photo_path="mock.jpg", seed=3)

    assert "OpenCV failed" in result["photo_analysis"]["error"]
    assert result["updated_features"]["Waist_circumference_cm"] == 90


def test_emotional_drift_is_exactly_seven_days_and_reproducible():
    first = analyze_emotional_drift(5, seed=42, return_details=True)
    second = analyze_emotional_drift(5, seed=42, return_details=True)

    assert first == second
    assert len(first["trajectory"]) == 7
    assert {"mean_stress", "final_stress", "volatility"} <= first.keys()


def test_pipeline_shapes_output_and_daily_hjb():
    result = run_pipeline(_features(), seed=7)

    assert set(result) == {
        "input_features",
        "updated_features",
        "photo_analysis",
        "emotional_drift",
        "local_latents",
        "bai",
        "physiological_state",
        "hjb",
        "weekly_plan",
        "warnings",
    }
    assert all(len(values) == 4 for values in result["local_latents"].values())
    assert sum(len(values) for values in result["local_latents"].values()) == 24
    assert len(result["bai"]) == 4
    assert all(
        0.0 <= value <= 100.0
        for value in result["physiological_state"]["scores"].values()
    )
    assert result["hjb"]["physiological_scores"] == {
        name: result["physiological_state"][name]
        for name in ("fatigue", "recovery", "stress", "readiness", "adaptation")
    }
    assert len(result["hjb"]["daily"]) == 7
    assert [row["stress"] for row in result["hjb"]["daily"]] == result[
        "emotional_drift"
    ]["trajectory"]


def test_hjb_consumes_physiological_scores():
    from ml.hjb.hjb_optimizer import optimize_hjb_trajectory

    base = {
        "daily_stress": [5.0] * 7,
        "hrv": 70.0,
        "sleep_hours": 7.0,
        "systolic_bp": 120.0,
        "crp_mg_l": 1.0,
        "fatigue": 10.0,
        "recovery": 90.0,
        "stress": 10.0,
        "readiness": 90.0,
        "adaptation": 90.0,
    }
    low = dict(base, fatigue=90.0, recovery=10.0, stress=90.0,
               readiness=10.0, adaptation=10.0)

    high_readiness = optimize_hjb_trajectory(base)
    low_readiness = optimize_hjb_trajectory(low)
    assert low_readiness["risk"] > high_readiness["risk"]
    assert low_readiness["optimal_intensity"] < high_readiness["optimal_intensity"]


def test_seed_reproduces_latents_and_bai():
    first = run_pipeline(_features(), seed=99)
    second = run_pipeline(_features(), seed=99)

    assert first["local_latents"] == second["local_latents"]
    assert first["bai"] == second["bai"]


def test_low_hrv_gate_and_critical_crp_override():
    low_hrv = run_pipeline(_features(HRV=30), seed=5)
    assert low_hrv["hjb"]["optimal_intensity"] <= 0.25
    assert low_hrv["weekly_plan"]["weekly_hiit_sessions"] == 0

    critical = run_pipeline(_features(CRP_mg_l=5), seed=5)
    assert critical["hjb"]["critical_crp_override"] is True
    assert critical["hjb"]["risk"] == 10.0
    assert critical["hjb"]["optimal_intensity"] == 0.0
    assert critical["weekly_plan"]["weekly_hiit_sessions"] == 0
    assert critical["weekly_plan"]["training_recommendation"] == "medical_review"


def test_generator_is_supported():
    result = run_pipeline(_features(), rng=np.random.default_rng(123))
    assert len(result["emotional_drift"]["trajectory"]) == 7


def test_mean_stress_updates_canonical_feature():
    result = run_pipeline(_features(), seed=13)
    assert result["updated_features"]["Emotional_stress"] == result[
        "emotional_drift"
    ]["mean_stress"]


def test_invalid_and_missing_features_are_rejected():
    missing = _features()
    missing.pop("HRV")
    try:
        run_pipeline(missing, seed=1)
    except ValueError as exc:
        assert "HRV" in str(exc)
    else:
        raise AssertionError("Missing HRV must be rejected")

    invalid = _features(Age=5)
    try:
        run_pipeline(invalid, seed=1)
    except ValueError as exc:
        assert "Age must be between" in str(exc)
    else:
        raise AssertionError("Out-of-range age must be rejected")


def test_emotional_drift_validates_simulation_arguments():
    for kwargs in ({"days": 0}, {"n_sim": 0}):
        try:
            analyze_emotional_drift(5, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {kwargs}")
