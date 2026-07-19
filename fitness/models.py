from django.db import models
from django.utils import timezone


class Athlete(models.Model):
    public_id = models.CharField(max_length=100, unique=True, db_index=True)
    name = models.CharField(max_length=150, blank=True)
    sex = models.CharField(max_length=20, blank=True)
    date_of_birth = models.DateField(null=True, blank=True)
    height_cm = models.FloatField(null=True, blank=True)
    sport_type = models.CharField(max_length=100, blank=True)
    training_level = models.CharField(max_length=50, blank=True)
    training_years = models.FloatField(null=True, blank=True)
    profile_data = models.JSONField(default=dict, blank=True)
    is_demonstration = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.public_id


class AthleteMeasurement(models.Model):
    athlete = models.ForeignKey(
        Athlete, on_delete=models.CASCADE, related_name="measurements"
    )
    measured_at = models.DateTimeField(default=timezone.now, db_index=True)
    source_type = models.CharField(max_length=50, default="manual_onboarding")
    source_dataset = models.ForeignKey(
        "DatasetRegistry", null=True, blank=True, on_delete=models.SET_NULL,
        related_name="measurements",
    )
    raw_inputs = models.JSONField(default=dict)
    raw_payload = models.JSONField(default=dict, blank=True)
    units = models.JSONField(default=dict)
    sources = models.JSONField(default=dict)
    input_quality = models.FloatField(default=0)
    missing_features = models.JSONField(default=list)
    imputed_features = models.JSONField(default=dict)
    measured_fields = models.JSONField(default=list, blank=True)
    reported_fields = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-measured_at"]


class BioenergeticAssessment(models.Model):
    athlete = models.ForeignKey(
        Athlete, on_delete=models.CASCADE, related_name="assessments"
    )
    measurement = models.ForeignKey(
        AthleteMeasurement, on_delete=models.PROTECT, related_name="assessments"
    )
    model_version = models.CharField(max_length=200)
    bai = models.FloatField()
    bai_normalized = models.FloatField()
    confidence = models.FloatField()
    uncertainty = models.FloatField()
    reconstruction_error = models.FloatField()
    latent_states = models.JSONField(default=dict)
    fatigue = models.FloatField()
    recovery = models.FloatField()
    stress = models.FloatField()
    inflammation = models.FloatField()
    performance = models.FloatField()
    readiness = models.FloatField()
    adaptation = models.FloatField()
    risk_flags = models.JSONField(default=list)
    explanation = models.JSONField(default=dict)
    result = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]


class Recommendation(models.Model):
    assessment = models.OneToOneField(
        BioenergeticAssessment,
        on_delete=models.CASCADE,
        related_name="recommendation_record",
    )
    training_intensity = models.FloatField()
    recovery_recommendation = models.TextField()
    hjb_action = models.JSONField(default=dict)
    safety_restrictions = models.JSONField(default=list)
    explanation = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


class DatasetRegistry(models.Model):
    DATASET_TYPES = [
        (value, value.title()) for value in (
            "athlete", "fitness", "running", "strength", "physiological",
            "wearable", "hrv", "recovery", "sleep", "laboratory",
        )
    ]
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    source_url = models.URLField(blank=True)
    license_name = models.CharField(max_length=200, blank=True)
    license_url = models.URLField(blank=True)
    dataset_type = models.CharField(max_length=30, choices=DATASET_TYPES)
    cohort_type = models.CharField(max_length=100, blank=True)
    participant_count = models.PositiveIntegerField(null=True, blank=True)
    record_count = models.PositiveIntegerField(default=0)
    feature_schema = models.JSONField(default=dict)
    target_schema = models.JSONField(default=dict, blank=True)
    feature_mapping = models.JSONField(default=dict, blank=True)
    local_file_path = models.CharField(max_length=500)
    imported_at = models.DateTimeField(null=True, blank=True)
    is_real_data = models.BooleanField(default=False)
    is_synthetic = models.BooleanField(default=False)
    limitations = models.TextField(blank=True)
    citation = models.TextField(blank=True)

    def __str__(self):
        return self.name


class ModelRun(models.Model):
    ARCHITECTURES = [("legacy", "Legacy FitAI"), ("bioenergetic", "Latent bioenergetic")]
    athlete = models.ForeignKey(Athlete, on_delete=models.CASCADE, related_name="model_runs")
    measurement = models.ForeignKey(AthleteMeasurement, on_delete=models.CASCADE, related_name="model_runs")
    architecture_type = models.CharField(max_length=30, choices=ARCHITECTURES)
    model_name = models.CharField(max_length=200)
    model_version = models.CharField(max_length=200, blank=True)
    executed_at = models.DateTimeField(auto_now_add=True)
    input_snapshot = models.JSONField(default=dict)
    output_snapshot = models.JSONField(default=dict)
    warnings = models.JSONField(default=list, blank=True)
    duration_ms = models.FloatField(default=0)


class AssessmentResult(models.Model):
    measurement = models.OneToOneField(
        AthleteMeasurement, on_delete=models.CASCADE, related_name="advanced_result"
    )
    assessment = models.OneToOneField(
        BioenergeticAssessment, null=True, blank=True, on_delete=models.SET_NULL,
        related_name="normalized_result",
    )
    bai = models.FloatField()
    confidence = models.FloatField()
    uncertainty = models.FloatField()
    physiological_states = models.JSONField(default=dict)
    latent_states = models.JSONField(default=dict)
    risk_flags = models.JSONField(default=list)
    recommendation = models.JSONField(default=dict)
    hjb_plan = models.JSONField(default=dict)
    explanation = models.JSONField(default=dict)
    limitations = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)


class ModelComparison(models.Model):
    measurement = models.OneToOneField(
        AthleteMeasurement, on_delete=models.CASCADE, related_name="comparison"
    )
    legacy_model_run = models.ForeignKey(
        ModelRun, on_delete=models.PROTECT, related_name="legacy_comparisons"
    )
    new_architecture_model_run = models.ForeignKey(
        ModelRun, on_delete=models.PROTECT, related_name="advanced_comparisons"
    )
    compatible_output_differences = models.JSONField(default=dict)
    added_capabilities = models.JSONField(default=list)
    stability_metrics = models.JSONField(default=dict)
    summary = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)


class EvaluationRun(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    dataset_count = models.PositiveIntegerField(default=0)
    total_records = models.PositiveIntegerField(default=0)
    valid_records = models.PositiveIntegerField(default=0)
    rejected_records = models.PositiveIntegerField(default=0)
    metrics = models.JSONField(default=dict)
    json_path = models.CharField(max_length=500, blank=True)
    csv_path = models.CharField(max_length=500, blank=True)

class UserData(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)

    # 23 параметри
    age = models.FloatField(null=True, blank=True)
    height_cm = models.FloatField(null=True, blank=True)
    weight_kg = models.FloatField(null=True, blank=True)
    waist_circumference_cm = models.FloatField(null=True, blank=True)
    emotional_stress = models.FloatField(null=True, blank=True)
    alcohol_units_per_week = models.FloatField(null=True, blank=True)
    daily_calories_kcal = models.FloatField(null=True, blank=True)
    max_push_ups = models.FloatField(null=True, blank=True)
    max_pull_ups = models.FloatField(null=True, blank=True)
    run_1km_min = models.FloatField(null=True, blank=True)
    run_100m_sec = models.FloatField(null=True, blank=True)
    cooper_test_km = models.FloatField(null=True, blank=True)
    burpees_3min = models.FloatField(null=True, blank=True)
    push_ups_1min = models.FloatField(null=True, blank=True)
    sleep_hours = models.FloatField(null=True, blank=True)
    resting_heart_rate_bpm = models.FloatField(null=True, blank=True)
    hrv = models.FloatField(null=True, blank=True)
    systolic_blood_pressure_mmhg = models.FloatField(null=True, blank=True)
    mitochondria_placeholder = models.FloatField(null=True, blank=True)
    testosterone_ng_dl = models.FloatField(null=True, blank=True)
    cortisol_ug_dl = models.FloatField(null=True, blank=True)
    hemoglobin_g_dl = models.FloatField(null=True, blank=True)
    crp_mg_l = models.FloatField(null=True, blank=True)

    # Результати
    prediction = models.JSONField(null=True, blank=True)
    weekly_plan = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"Data {self.timestamp.date()}"
