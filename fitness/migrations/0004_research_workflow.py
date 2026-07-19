import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("fitness", "0003_athlete_assessment_models")]
    operations = [
        migrations.CreateModel(name="DatasetRegistry", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("name", models.CharField(max_length=200, unique=True)), ("description", models.TextField(blank=True)),
            ("source_url", models.URLField(blank=True)), ("license_name", models.CharField(blank=True, max_length=200)),
            ("license_url", models.URLField(blank=True)),
            ("dataset_type", models.CharField(choices=[("athlete", "Athlete"), ("fitness", "Fitness"), ("running", "Running"), ("strength", "Strength"), ("physiological", "Physiological"), ("wearable", "Wearable"), ("hrv", "Hrv"), ("recovery", "Recovery"), ("sleep", "Sleep"), ("laboratory", "Laboratory")], max_length=30)),
            ("cohort_type", models.CharField(blank=True, max_length=100)), ("participant_count", models.PositiveIntegerField(blank=True, null=True)),
            ("record_count", models.PositiveIntegerField(default=0)), ("feature_schema", models.JSONField(default=dict)),
            ("target_schema", models.JSONField(blank=True, default=dict)), ("feature_mapping", models.JSONField(blank=True, default=dict)),
            ("local_file_path", models.CharField(max_length=500)), ("imported_at", models.DateTimeField(blank=True, null=True)),
            ("is_real_data", models.BooleanField(default=False)), ("is_synthetic", models.BooleanField(default=False)),
            ("limitations", models.TextField(blank=True)), ("citation", models.TextField(blank=True)),
        ]),
        migrations.CreateModel(name="EvaluationRun", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("created_at", models.DateTimeField(auto_now_add=True)), ("dataset_count", models.PositiveIntegerField(default=0)),
            ("total_records", models.PositiveIntegerField(default=0)), ("valid_records", models.PositiveIntegerField(default=0)),
            ("rejected_records", models.PositiveIntegerField(default=0)), ("metrics", models.JSONField(default=dict)),
            ("json_path", models.CharField(blank=True, max_length=500)), ("csv_path", models.CharField(blank=True, max_length=500)),
        ]),
        migrations.AddField(model_name="athlete", name="date_of_birth", field=models.DateField(blank=True, null=True)),
        migrations.AddField(model_name="athlete", name="height_cm", field=models.FloatField(blank=True, null=True)),
        migrations.AddField(model_name="athlete", name="name", field=models.CharField(blank=True, max_length=150)),
        migrations.AddField(model_name="athletemeasurement", name="measured_fields", field=models.JSONField(blank=True, default=list)),
        migrations.AddField(model_name="athletemeasurement", name="raw_payload", field=models.JSONField(blank=True, default=dict)),
        migrations.AddField(model_name="athletemeasurement", name="reported_fields", field=models.JSONField(blank=True, default=list)),
        migrations.AddField(model_name="athletemeasurement", name="source_type", field=models.CharField(default="manual_onboarding", max_length=50)),
        migrations.CreateModel(name="AssessmentResult", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("bai", models.FloatField()), ("confidence", models.FloatField()), ("uncertainty", models.FloatField()),
            ("physiological_states", models.JSONField(default=dict)), ("latent_states", models.JSONField(default=dict)),
            ("risk_flags", models.JSONField(default=list)), ("recommendation", models.JSONField(default=dict)),
            ("hjb_plan", models.JSONField(default=dict)), ("explanation", models.JSONField(default=dict)),
            ("limitations", models.JSONField(default=list)), ("created_at", models.DateTimeField(auto_now_add=True)),
            ("assessment", models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="normalized_result", to="fitness.bioenergeticassessment")),
            ("measurement", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="advanced_result", to="fitness.athletemeasurement")),
        ]),
        migrations.AddField(model_name="athletemeasurement", name="source_dataset", field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="measurements", to="fitness.datasetregistry")),
        migrations.CreateModel(name="ModelRun", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("architecture_type", models.CharField(choices=[("legacy", "Legacy FitAI"), ("bioenergetic", "Latent bioenergetic")], max_length=30)),
            ("model_name", models.CharField(max_length=200)), ("model_version", models.CharField(blank=True, max_length=200)),
            ("executed_at", models.DateTimeField(auto_now_add=True)), ("input_snapshot", models.JSONField(default=dict)),
            ("output_snapshot", models.JSONField(default=dict)), ("warnings", models.JSONField(blank=True, default=list)), ("duration_ms", models.FloatField(default=0)),
            ("athlete", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="model_runs", to="fitness.athlete")),
            ("measurement", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="model_runs", to="fitness.athletemeasurement")),
        ]),
        migrations.CreateModel(name="ModelComparison", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("compatible_output_differences", models.JSONField(default=dict)), ("added_capabilities", models.JSONField(default=list)),
            ("stability_metrics", models.JSONField(default=dict)), ("summary", models.TextField()), ("created_at", models.DateTimeField(auto_now_add=True)),
            ("measurement", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="comparison", to="fitness.athletemeasurement")),
            ("legacy_model_run", models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name="legacy_comparisons", to="fitness.modelrun")),
            ("new_architecture_model_run", models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name="advanced_comparisons", to="fitness.modelrun")),
        ]),
    ]
