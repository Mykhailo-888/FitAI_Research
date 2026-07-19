"""Run a reproducible, explicitly fictional FitAI athlete assessment."""
import argparse
import json
import os
from pathlib import Path


DEMO = {
    "athlete_id": "demo-athlete-001", "is_demonstration": True,
    "sex": "male", "sport_type": "hybrid endurance/strength",
    "training_level": "intermediate", "training_years": 4,
    "age": 29, "height_cm": 181, "weight_kg": 79, "waist_cm": 82,
    "emotional_stress": 4, "alcohol_units_per_week": 2,
    "daily_calories_kcal": 2800, "max_push_ups": 48, "max_pull_ups": 13,
    "run_1km_min": 4.25, "run_100m_sec": 13.8, "cooper_test_km": 3.05,
    "burpees_3min": 58, "push_ups_1min": 42, "sleep_hours": 7.6,
    "resting_heart_rate_bpm": 54, "systolic_blood_pressure_mmhg": 118,
    "mitochondria_placeholder": None, "testosterone_ng_dl": 610,
    "cortisol_ug_dl": 13.2, "hemoglobin_g_dl": 15.1, "crp_mg_l": 0.9, "hrv": 72,
    "sources": {
        "Sleep_hours": "athlete-reported", "Emotional_stress": "athlete-reported",
        "Alcohol_units_per_week": "athlete-reported",
        "Resting_heart_rate_bpm": "measured", "Systolic_blood_pressure_mmhg": "measured",
        "Testosterone_ng_dl": "measured", "Cortisol_ug_dl": "measured",
        "Hemoglobin_g_dl": "measured", "CRP_mg_l": "measured", "HRV": "measured",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Optional athlete JSON input")
    parser.add_argument("--output", type=Path, default=Path("assessment_output.json"))
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8")) if args.input else DEMO
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fitai.settings")
    import django
    django.setup()
    from ml.services.athlete_assessment import assess_athlete
    result = assess_athlete(payload, persist=not args.no_persist)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    b, s = result["bioenergetic_state"], result["physiological_states"]
    print(f"FitAI assessment: {result['athlete_id']} (demonstration data)")
    print(f"BAI: {b['bai_normalized']:.1f}/100 | confidence: {b['confidence']:.0%}")
    print(f"Recovery {s['recovery']:.1f} | Stress {s['stress']:.1f} | Readiness {s['readiness']:.1f} | Adaptation {s['adaptation']:.1f}")
    print(f"Action: {result['recommendation']['action']} ({result['recommendation']['reason']})")
    print(f"Personal baseline: {result['personal_baseline']['status']}")
    print(f"Saved JSON: {args.output.resolve()}")
    print(result["disclaimer"])


if __name__ == "__main__":
    main()
