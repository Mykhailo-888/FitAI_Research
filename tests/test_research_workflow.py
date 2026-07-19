import pytest
from django.urls import reverse

from fitness.dataset_service import inspect_and_register_local_datasets, map_row
from fitness.models import Athlete, DatasetRegistry, ModelComparison
from fitness.services import assess_and_compare
from ml.demo_athlete_assessment import DEMO


@pytest.mark.django_db
def test_dataset_registration_and_mapping():
    datasets = inspect_and_register_local_datasets()
    assert len(datasets) == 3
    real = DatasetRegistry.objects.get(name="gym_members_exercise_tracking")
    assert real.record_count == 973
    assert real.license_name == "Apache License 2.0"
    mapped, missing = map_row(real, {"Age": 30, "Weight (kg)": 80, "Height (m)": 1.8, "Resting_BPM": 60})
    assert mapped["Height_cm"] == 180
    assert len(missing) == 19


@pytest.mark.django_db
def test_identical_inputs_dual_runs_and_persistence():
    comparison = assess_and_compare(DEMO)
    assert comparison.measurement.model_runs.count() == 2
    assert comparison.legacy_model_run.input_snapshot == comparison.new_architecture_model_run.input_snapshot
    assert all(value == 0 for value in comparison.compatible_output_differences.values())
    assert comparison.measurement.advanced_result.recommendation["code"] in {
        "TRAIN", "TRAIN_LIGHT", "ACTIVE_RECOVERY", "REST", "MEDICAL_CAUTION"
    }


@pytest.mark.django_db
def test_multiple_measurements_personal_history_and_results_render(client):
    first = assess_and_compare(dict(DEMO, athlete_id="history-person"))
    second = assess_and_compare(dict(DEMO, athlete_id="history-person", sleep_hours=6.8))
    athlete = Athlete.objects.get(public_id="history-person")
    assert athlete.measurements.count() == 2
    response = client.get(reverse("assessment_results", args=[second.pk]))
    assert response.status_code == 200
    assert b"Old vs new architecture" in response.content
    history = client.get(reverse("athlete_history", args=[athlete.public_id]))
    assert history.status_code == 200
    assert first.pk != second.pk


@pytest.mark.django_db
def test_crp_gate_and_train_recommendations():
    caution = assess_and_compare(dict(DEMO, athlete_id="crp-person", crp_mg_l=8))
    assert caution.new_architecture_model_run.output_snapshot["recommendation"]["code"] == "MEDICAL_CAUTION"
    healthy = assess_and_compare(dict(
        DEMO, athlete_id="train-person", sleep_hours=9, hrv=110,
        emotional_stress=1, crp_mg_l=.2,
    ))
    assert healthy.new_architecture_model_run.output_snapshot["recommendation"]["code"] in {"TRAIN", "TRAIN_LIGHT"}


@pytest.mark.django_db
def test_dataset_registry_page(client):
    inspect_and_register_local_datasets()
    response = client.get(reverse("dataset_registry"))
    assert response.status_code == 200
    assert b"Apache License 2.0" in response.content
