# Dataset registry and provenance

The executable registry is `ml/data/dataset_registry.py`.

| Internal name | Status | Use | Limitations |
|---|---|---|---|
| `fitai_engineered_23` | derived/synthetic | explicit imputation medians | not clinical or a population reference |
| `fitai_extended_latent` | derived/synthetic | retained research artifact | generated latents are not measurements |
| `gym_members_exercise_tracking` | externally sourced | partial age/height/weight/resting-BPM comparisons | exact URL, author and license absent |

The service returns `partial_reference_data`, cohort size and limitations. No external
data were downloaded, and no citations or rows were invented. The README identifies
the gym data as Kaggle-sourced, but exact provenance and licensing must be restored
before treating it as a verified scientific cohort.
