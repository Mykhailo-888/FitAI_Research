"""Trajectory-aware adapter around FitAI's legacy HJB-inspired optimizer."""

from collections.abc import Mapping

from ml.training_risk import predict_risk


CRITICAL_CRP_MG_L = 5.0
LOW_HRV = 50.0


def optimize_hjb_trajectory(
    physiological_state: Mapping,
    *,
    predict_fn=predict_risk,
) -> dict:
    """Evaluate the trajectory using BAI physiology and repository safety gates."""
    trajectory = list(physiological_state["daily_stress"])
    if len(trajectory) != 7:
        raise ValueError("HJB requires exactly 7 daily stress values")
    score_names = ("fatigue", "recovery", "stress", "readiness", "adaptation")
    try:
        scores = {
            name: float(physiological_state[name]) for name in score_names
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "HJB requires numeric physiological-state scores"
        ) from exc
    if any(not 0.0 <= value <= 100.0 for value in scores.values()):
        raise ValueError("HJB physiological-state scores must be in [0, 100]")

    daily = []
    for day, stress in enumerate(trajectory, start=1):
        risk, text, intensity = predict_fn(
            physiological_state["hrv"],
            physiological_state["sleep_hours"],
            physiological_state["systolic_bp"],
            stress,
            physiological_state["crp_mg_l"],
            days=1,
        )
        daily.append(
            {
                "day": day,
                "stress": float(stress),
                "risk": float(risk),
                "optimal_intensity": float(intensity),
                "recommendation": text,
            }
        )

    risk = max(item["risk"] for item in daily)
    optimal_intensity = min(item["optimal_intensity"] for item in daily)
    warnings = []

    physiological_load = max(scores["fatigue"], scores["stress"]) / 100.0
    physiological_capacity = min(
        scores["recovery"], scores["readiness"], scores["adaptation"]
    ) / 100.0
    risk = min(10.0, risk + 2.0 * physiological_load)
    optimal_intensity = min(optimal_intensity, physiological_capacity)
    if physiological_capacity < 0.5:
        warnings.append("Low BAI physiological readiness gate applied.")

    if float(physiological_state["hrv"]) < LOW_HRV:
        optimal_intensity = min(optimal_intensity, 0.25)
        warnings.append("Low HRV safety gate applied.")

    critical_override = (
        float(physiological_state["crp_mg_l"]) >= CRITICAL_CRP_MG_L
    )
    if critical_override:
        risk = 10.0
        optimal_intensity = 0.0
        warnings.append(
            "Critical CRP safety override: pause training and obtain "
            "qualified clinical guidance."
        )

    return {
        "risk": float(risk),
        "optimal_intensity": float(optimal_intensity),
        "daily": daily,
        "physiological_scores": scores,
        "critical_crp_override": critical_override,
        "warnings": warnings,
        "decision_support_only": True,
    }
