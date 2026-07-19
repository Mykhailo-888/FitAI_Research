# fitness/admin.py

from django.contrib import admin
from .models import Athlete, AthleteMeasurement, BioenergeticAssessment, Recommendation

admin.site.register(Athlete)
admin.site.register(AthleteMeasurement)
admin.site.register(BioenergeticAssessment)
admin.site.register(Recommendation)
from .models import UserData  # тільки UserData

admin.site.register(UserData)
