from django.contrib import admin

from .models import (
    AssessmentResult, Athlete, AthleteMeasurement, BioenergeticAssessment,
    DatasetRegistry, EvaluationRun, ModelComparison, ModelRun,
    Recommendation, UserData,
)

admin.site.register([
    Athlete, AthleteMeasurement, BioenergeticAssessment, Recommendation,
    DatasetRegistry, ModelRun, AssessmentResult, ModelComparison,
    EvaluationRun, UserData,
])
