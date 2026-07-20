# FitAI research workflow

## Stable architecture entry points

- Legacy trained predictor: `fitness.services.run_legacy_analysis`, extracted from
  the original `fitness.views.process_onboarding_results` use of
  `ml.fit_model_core.get_fitness_model("simple").predict(...)`.
- Advanced assessment: `ml.services.athlete_assessment.assess_athlete`, the callable
  used by `python -m ml.demo_athlete_assessment`.
- Dual orchestration: `fitness.services.assess_and_compare`. It validates once,
  stores one canonical measurement, gives both architectures the identical
  23-feature snapshot, and persists their unmodified raw outputs.

The old `UserData` table and stable advanced assessment service remain available.
The former inline onboarding analysis below the service redirect is obsolete and
unreachable; it is retained temporarily to minimize destructive refactoring.

## Data and scientific scope

See `docs/datasets.md`. The only documented real dataset is Vala Khorasani's Gym
Members Exercise Dataset (973 records, Apache 2.0). It has no participant ID and
maps only four FitAI features, so it is a partial population reference and is not
passed through the 23-feature comparison with fabricated values. The two compatible
973-row datasets are derived/synthetic.

BAI is a research proxy, not a direct mitochondrial measurement. Latent states are
model-derived and proof-of-concept because trained hierarchical VAE weights are
absent. Richer outputs are not proof of predictive superiority. Recommendations
are research decision support, not diagnosis or treatment.

## Persistence

Migration `fitness/0004_research_workflow.py` adds `DatasetRegistry`,
`EvaluationRun`, `ModelRun`, `AssessmentResult`, and `ModelComparison`; extends
`Athlete`; and adds source/quality provenance to `AthleteMeasurement`. The existing
`BioenergeticAssessment`, `Recommendation`, and `UserData` models are preserved.

## Commands

```powershell
Copy-Item .env.example .env
# replace placeholder secrets in .env
docker compose up -d --build
docker compose exec web python manage.py database_info
docker compose exec web python manage.py evaluate_architectures
```

Local isolated tests may use SQLite:

```powershell
python manage.py check
python manage.py makemigrations --check
python manage.py migrate
python -m pytest -q
```

Evaluation JSON and CSV are written to `evaluation_results/` and the same summary
is persisted in `EvaluationRun`.

## Browser routes

- Onboarding: `http://localhost:8003/onboarding/`
- Results: `http://localhost:8003/results/<comparison_id>/`
- Athlete history: `http://localhost:8003/athletes/<athlete_id>/history/`
- Architecture comparison: `http://localhost:8003/comparisons/<comparison_id>/`
- Dataset registry: `http://localhost:8003/datasets/`

## Docker verification note

The PostgreSQL container initialized successfully and reported PostgreSQL 16.14,
and Compose reported it healthy. On the verification workstation, Docker Desktop
repeatedly timed out rebuilding the web image, then stopped responding during a
web-container restart. Therefore current-source container migration and HTTP proof
must be rerun with the commands above on a healthy Docker installation. Do not
interpret the local SQLite test run as final PostgreSQL integration proof.
