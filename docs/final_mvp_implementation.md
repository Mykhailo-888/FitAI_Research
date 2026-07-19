# Final MVP implementation

## Architecture actually found

The canonical input is the 23-column order in `ml.bioenergetics.latent_states.FEATURE_INDEX`.
The persisted trained artifact is `ml/models/trained_fitness_model_simple.pkl`, a custom
NumPy 23-input, 8-output network. The repository also contains six 4-dimensional VAE
encoders, a 24-dimensional concatenated vector, a 4-dimensional Bioenergetic Core,
and a 23-feature decoder.

No serialized hierarchical VAE state was found. That layer is therefore a deterministic
proof of concept, not a trained estimator. Output says so, caps confidence, and uses
the trained legacy model alongside it. No model artifact was changed.

`ml.services.athlete_assessment` implements:

athlete JSON -> validation/unit/source tracking -> explicit dataset-median imputation
-> trained model plus research hierarchy -> interpretable states -> reference/personal
comparison -> HJB-inspired safety control -> persistence/report.

The existing `UserData` flow remains for backward compatibility.

## Canonical schema

Order: Age, Height_cm, Weight_kg, Waist_circumference_cm, Emotional_stress,
Alcohol_units_per_week, Daily_calories_kcal, Max_push_ups, Max_pull_ups,
Run_1km_min, Run_100m_sec, Cooper_test_km, Burpees_3min, Push_ups_1min,
Sleep_hours, Resting_heart_rate_bpm, Systolic_blood_pressure_mmhg,
Mitochondria_placeholder, Testosterone_ng_dl, Cortisol_ug_dl, Hemoglobin_g_dl,
CRP_mg_l, HRV.

Units and plausible bounds are declared in the assessment service. Missing values
use medians from the derived 23-feature dataset and every imputation is returned.
The mitochondria placeholder is a legacy unitless proxy, never a measurement.

## BAI, states, and confidence

Raw BAI is the mean of the four exchangeable global latent coordinates. Its 0-100
display score uses the repository's symmetric `PhysiologyState` transform; it is
not a clinical percentile. Both values are stored. Confidence combines completeness,
measured-source fraction and reference availability, with a mandatory penalty for
the missing trained hierarchy.

All states are bounded 0-100. Higher recovery, performance, readiness and adaptation
are favorable; higher fatigue, stress and inflammation are unfavorable. CRP, low HRV,
systolic pressure, and severe fatigue/stress can restrict training. The HJB-inspired
output is decision support, not a clinically optimal policy.

## Database and verification

Migration `fitness.0003_athlete_assessment_models` adds Athlete, AthleteMeasurement,
BioenergeticAssessment and Recommendation. Query fields are explicit columns; raw
measurements, units, sources, vectors and explanations use JSON.

PostgreSQL is selected by `DATABASE_URL` or `POSTGRES_*`. SQLite is a development/test
fallback with `DEBUG=True`.

Python compilation, Django checks, migration drift checks, SQLite migration, trained
artifact loading, the persistent demo, and 49 tests ran successfully on 2026-07-19.
A live PostgreSQL server was unavailable, so live PostgreSQL connectivity was not verified.
