from django.db import migrations, models


INPUT_FIELDS = (
    'age', 'height_cm', 'weight_kg', 'waist_circumference_cm',
    'emotional_stress', 'alcohol_units_per_week', 'daily_calories_kcal',
    'max_push_ups', 'max_pull_ups', 'run_1km_min', 'run_100m_sec',
    'cooper_test_km', 'burpees_3min', 'push_ups_1min', 'sleep_hours',
    'resting_heart_rate_bpm', 'systolic_blood_pressure_mmhg',
    'mitochondria_placeholder', 'testosterone_ng_dl', 'cortisol_ug_dl',
    'hemoglobin_g_dl', 'crp_mg_l', 'hrv',
)


def separate_legacy_stress(apps, schema_editor):
    UserData = apps.get_model('fitness', 'UserData')
    rows = UserData.objects.using(schema_editor.connection.alias)

    for row in rows.iterator(chunk_size=500):
        sources = {name: 'legacy_unknown' for name in INPUT_FIELDS}

        # Зберігаємо відомі джерела з попереднього пакета.
        if isinstance(row.weekly_plan, dict):
            previous = row.weekly_plan.get('input_sources', {})
            if isinstance(previous, dict):
                for name in INPUT_FIELDS:
                    if previous.get(name) in ('reported', 'default'):
                        sources[name] = previous[name]

        sources['emotional_stress'] = 'legacy_forecast_only'

        # Раніше це поле містило прогноз, а не початковий стрес.
        row.predicted_stress_7d = row.emotional_stress
        row.emotional_stress = None
        row.input_sources = sources

        row.save(
            using=schema_editor.connection.alias,
            update_fields=[
                'emotional_stress',
                'predicted_stress_7d',
                'input_sources',
            ],
        )


def restore_legacy_stress(apps, schema_editor):
    UserData = apps.get_model('fitness', 'UserData')
    rows = UserData.objects.using(schema_editor.connection.alias)

    for row in rows.iterator(chunk_size=500):
        if row.predicted_stress_7d is not None:
            row.emotional_stress = row.predicted_stress_7d
            row.save(
                using=schema_editor.connection.alias,
                update_fields=['emotional_stress'],
            )


class Migration(migrations.Migration):
    dependencies = [
        ('fitness', '0002_remove_userdata_profile_alter_userdata_options_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='userdata',
            name='predicted_stress_7d',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='userdata',
            name='input_sources',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.RunPython(
            separate_legacy_stress,
            restore_legacy_stress,
        ),
    ]
