import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("fitness", "0002_remove_userdata_profile_alter_userdata_options_and_more")]

    operations = [
        migrations.CreateModel(
            name="Athlete",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("public_id", models.CharField(db_index=True, max_length=100, unique=True)),
                ("sex", models.CharField(blank=True, max_length=20)),
                ("sport_type", models.CharField(blank=True, max_length=100)),
                ("training_level", models.CharField(blank=True, max_length=50)),
                ("training_years", models.FloatField(blank=True, null=True)),
                ("profile_data", models.JSONField(blank=True, default=dict)),
                ("is_demonstration", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name="AthleteMeasurement",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("measured_at", models.DateTimeField(db_index=True, default=django.utils.timezone.now)),
                ("raw_inputs", models.JSONField(default=dict)),
                ("units", models.JSONField(default=dict)),
                ("sources", models.JSONField(default=dict)),
                ("input_quality", models.FloatField(default=0)),
                ("missing_features", models.JSONField(default=list)),
                ("imputed_features", models.JSONField(default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("athlete", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="measurements", to="fitness.athlete")),
            ],
            options={"ordering": ["-measured_at"]},
        ),
        migrations.CreateModel(
            name="BioenergeticAssessment",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("model_version", models.CharField(max_length=200)),
                ("bai", models.FloatField()), ("bai_normalized", models.FloatField()),
                ("confidence", models.FloatField()), ("uncertainty", models.FloatField()),
                ("reconstruction_error", models.FloatField()), ("latent_states", models.JSONField(default=dict)),
                ("fatigue", models.FloatField()), ("recovery", models.FloatField()),
                ("stress", models.FloatField()), ("inflammation", models.FloatField()),
                ("performance", models.FloatField()), ("readiness", models.FloatField()),
                ("adaptation", models.FloatField()), ("risk_flags", models.JSONField(default=list)),
                ("explanation", models.JSONField(default=dict)), ("result", models.JSONField(default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                ("athlete", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="assessments", to="fitness.athlete")),
                ("measurement", models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name="assessments", to="fitness.athletemeasurement")),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.CreateModel(
            name="Recommendation",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("training_intensity", models.FloatField()),
                ("recovery_recommendation", models.TextField()),
                ("hjb_action", models.JSONField(default=dict)),
                ("safety_restrictions", models.JSONField(default=list)),
                ("explanation", models.TextField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("assessment", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="recommendation_record", to="fitness.bioenergeticassessment")),
            ],
        ),
    ]
