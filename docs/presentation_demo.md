# 3-5 minute presentation demonstration

1. **Purpose (30 seconds).** FitAI estimates an athlete-specific adaptation state
   for research decision support; BAI is not a mitochondrial measurement or diagnosis.
2. **Input (45 seconds).** Show the fictional profile in `ml/demo_athlete_assessment.py`,
   its source metadata, and the intentionally missing mitochondria placeholder.
3. **Database (30 seconds).** Explain Athlete -> Measurement -> Assessment ->
   Recommendation.
4. **Assessment (60 seconds).** Run
   `python -m ml.demo_athlete_assessment --output demo_assessment_output.json`.
   Show BAI, confidence, six latent summaries, states and imputation details.
5. **Safety/control (45 seconds).** Show HJB-inspired intensity. Explain CRP, HRV,
   pressure, fatigue and stress gates.
6. **Comparisons (30 seconds).** Show partial gym-member comparison and
   `insufficient_history`; three prior assessments enable the personal median.
7. **Limitations (30 seconds).** The trained legacy artifact loads, but no saved
   hierarchical checkpoint exists. End with the disclaimer.

For a safety example, copy the demo profile to JSON, set `crp_mg_l` to 8, and run:

```powershell
python -m ml.demo_athlete_assessment --input athlete.json --no-persist
```

Training intensity should be gated to zero.
