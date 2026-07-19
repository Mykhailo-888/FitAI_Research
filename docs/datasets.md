# Dataset registry and provenance

The executable registry is `ml/data/dataset_registry.py`.

| Internal name | Status | Use | Limitations |
|---|---|---|---|
| `fitai_engineered_23` | derived/synthetic | explicit imputation medians | not clinical or a population reference |
| `fitai_extended_latent` | derived/synthetic | retained research artifact | generated latents are not measurements |
| `gym_members_exercise_tracking` | real gym-member session records, Apache-2.0 | partial age/height/weight/resting-BPM reference | no participant ID; not professional athletes; only 4/23 direct mappings |

The source is Vala Khorasani's 2024 [Gym Members Exercise Dataset on Kaggle](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset),
listed there under Apache License 2.0. The local file has 973 records and 15 fields.
It does not include a participant identifier, so participant count is unknown and
rows must not be treated as repeat measurements or unique identities. It is a gym
member dataset, not a professional-athlete cohort.

Direct mappings are `Age → Age`, `Weight (kg) → Weight_kg`,
`Height (m) → Height_cm` (unit conversion), and
`Resting_BPM → Resting_heart_rate_bpm`. The remaining 19 FitAI inputs are missing.
Consequently the dataset is used only as a partial cohort reference and is rejected
from full 23-feature architecture evaluation rather than silently imputing 19 fields.

The two `edited_23_params_*` files each contain 973 derived/synthetic rows. They are
compatible with the model but are not real-athlete validation data. The project has
no compatible real-data ground truth, so it does not publish accuracy, MAE, RMSE,
R², classification, or calibration claims.
