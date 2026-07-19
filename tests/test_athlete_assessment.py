import copy
import math

import pytest

from ml.demo_athlete_assessment import DEMO
from ml.pipeline import CANONICAL_FEATURES
from ml.services.athlete_assessment import assess_athlete, validate_athlete_input


def test_canonical_order_and_imputation_reporting():
    features, quality = validate_athlete_input({"age": 30})
    assert tuple(features) == CANONICAL_FEATURES
    assert "HRV" in quality["missing_features"]
    assert quality["imputed_features"]["HRV"]["source"] == "dataset-imputed"


def test_units_and_ranges_are_explicit():
    with pytest.raises(ValueError, match="requires unit"):
        validate_athlete_input({"age": 30, "units": {"Age": "months"}})
    with pytest.raises(ValueError, match="between"):
        validate_athlete_input({"age": 4})


def test_assessment_is_finite_bounded_and_deterministic():
    first = assess_athlete(DEMO)
    second = assess_athlete(DEMO)
    assert first["bioenergetic_state"]["bai_raw"] == second["bioenergetic_state"]["bai_raw"]
    assert math.isfinite(first["bioenergetic_state"]["bai_raw"])
    assert all(0 <= value <= 100 for value in first["physiological_states"].values())


def test_low_hrv_and_poor_sleep_reduce_readiness():
    good = assess_athlete(DEMO)
    poor = copy.deepcopy(DEMO)
    poor.update({"hrv": 30, "sleep_hours": 4.5})
    result = assess_athlete(poor)
    assert result["physiological_states"]["readiness"] < good["physiological_states"]["readiness"]
    assert result["hjb_control"]["optimal_intensity"] <= 0.25


def test_high_crp_safety_gate():
    payload = dict(DEMO, crp_mg_l=8)
    result = assess_athlete(payload)
    assert result["hjb_control"]["critical_crp_override"]
    assert result["recommendation"]["training_intensity"] == 0


def test_higher_quality_has_lower_uncertainty():
    sparse = assess_athlete({"athlete_id": "sparse", "age": 30})
    complete = assess_athlete(DEMO)
    assert complete["bioenergetic_state"]["uncertainty"] < sparse["bioenergetic_state"]["uncertainty"]


@pytest.mark.django_db
def test_persistence_and_first_baseline():
    result = assess_athlete(DEMO, persist=True)
    from fitness.models import Athlete, AthleteMeasurement, BioenergeticAssessment, Recommendation
    athlete = Athlete.objects.get(public_id=DEMO["athlete_id"])
    assert AthleteMeasurement.objects.filter(athlete=athlete).exists()
    assert BioenergeticAssessment.objects.filter(athlete=athlete).exists()
    assert Recommendation.objects.filter(assessment__athlete=athlete).exists()
    assert result["personal_baseline"]["status"] == "insufficient_history"


@pytest.mark.django_db
def test_personal_baseline_after_history():
    payload = dict(DEMO, athlete_id="baseline-athlete")
    for _ in range(3):
        assess_athlete(payload, persist=True)
    result = assess_athlete(payload)
    assert result["personal_baseline"]["status"] == "available"
