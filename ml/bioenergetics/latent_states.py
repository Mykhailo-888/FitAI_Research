"""
FitAI Physiological Hierarchical VAE

latent_states.py

Definition of physiological latent states and
feature grouping for the Hierarchical VAE.

Author: Mykhailo Velychko
"""

from ml.vae.encoder import VariationalEncoder
from ml.bioenergetics.local_decoders import LocalDecoder


# ============================================================
# Feature indices
# ============================================================

FEATURE_INDEX = {

    "Age": 0,
    "Height_cm": 1,
    "Weight_kg": 2,
    "Waist_circumference_cm": 3,
    "Emotional_stress": 4,
    "Alcohol_units_per_week": 5,
    "Daily_calories_kcal": 6,
    "Max_push_ups": 7,
    "Max_pull_ups": 8,
    "Run_1km_min": 9,
    "Run_100m_sec": 10,
    "Cooper_test_km": 11,
    "Burpees_3min": 12,
    "Push_ups_1min": 13,
    "Sleep_hours": 14,
    "Resting_heart_rate_bpm": 15,
    "Systolic_blood_pressure_mmhg": 16,
    "Mitochondria_placeholder": 17,
    "Testosterone_ng_dl": 18,
    "Cortisol_ug_dl": 19,
    "Hemoglobin_g_dl": 20,
    "CRP_mg_l": 21,
    "HRV": 22

}


# ============================================================
# Local physiological latent spaces
# ============================================================

ENERGY_FEATURES = [

    "Daily_calories_kcal",
    "HRV",
    "Resting_heart_rate_bpm",
    "Cooper_test_km",
    "Run_1km_min",
    "Mitochondria_placeholder"

]

RECOVERY_FEATURES = [

    "Sleep_hours",
    "HRV",
    "Resting_heart_rate_bpm",
    "Alcohol_units_per_week",
    "Cortisol_ug_dl"

]

STRESS_FEATURES = [

    "Emotional_stress",
    "HRV",
    "Cortisol_ug_dl",
    "CRP_mg_l",
    "Sleep_hours"

]

MUSCLE_FEATURES = [

    "Weight_kg",
    "Max_push_ups",
    "Max_pull_ups",
    "Burpees_3min",
    "Push_ups_1min",
    "Testosterone_ng_dl"

]

METABOLISM_FEATURES = [

    "Waist_circumference_cm",
    "Daily_calories_kcal",
    "Alcohol_units_per_week",
    "Hemoglobin_g_dl",
    "CRP_mg_l"

]

AGING_FEATURES = [

    "Age",
    "HRV",
    "Resting_heart_rate_bpm",
    "Systolic_blood_pressure_mmhg",
    "Hemoglobin_g_dl"

]


# ============================================================
# Helper
# ============================================================

def feature_indices(feature_names):

    return [

        FEATURE_INDEX[name]

        for name in feature_names

    ]


# ============================================================
# Hierarchical Encoders
# ============================================================

class HierarchicalLatentStates:

    def __init__(self):

        # ===================================================
        # Encoders
        # ===================================================

        self.energy_encoder = VariationalEncoder(
            input_size=len(ENERGY_FEATURES),
            hidden_size=16,
            latent_size=4
        )

        self.recovery_encoder = VariationalEncoder(
            input_size=len(RECOVERY_FEATURES),
            hidden_size=16,
            latent_size=4
        )

        self.stress_encoder = VariationalEncoder(
            input_size=len(STRESS_FEATURES),
            hidden_size=16,
            latent_size=4
        )

        self.muscle_encoder = VariationalEncoder(
            input_size=len(MUSCLE_FEATURES),
            hidden_size=16,
            latent_size=4
        )

        self.metabolism_encoder = VariationalEncoder(
            input_size=len(METABOLISM_FEATURES),
            hidden_size=16,
            latent_size=4
        )

        self.aging_encoder = VariationalEncoder(
            input_size=len(AGING_FEATURES),
            hidden_size=16,
            latent_size=4
        )

        # ===================================================
        # Local Decoders
        # ===================================================

        self.energy_decoder = LocalDecoder(
            latent_size=4,
            hidden_size=16,
            output_size=len(ENERGY_FEATURES)
        )

        self.recovery_decoder = LocalDecoder(
            latent_size=4,
            hidden_size=16,
            output_size=len(RECOVERY_FEATURES)
        )

        self.stress_decoder = LocalDecoder(
            latent_size=4,
            hidden_size=16,
            output_size=len(STRESS_FEATURES)
        )

        self.muscle_decoder = LocalDecoder(
            latent_size=4,
            hidden_size=16,
            output_size=len(MUSCLE_FEATURES)
        )

        self.metabolism_decoder = LocalDecoder(
            latent_size=4,
            hidden_size=16,
            output_size=len(METABOLISM_FEATURES)
        )

        self.aging_decoder = LocalDecoder(
            latent_size=4,
            hidden_size=16,
            output_size=len(AGING_FEATURES)
        )

    def forward(self, x):

        energy = x[:, feature_indices(ENERGY_FEATURES)]
        recovery = x[:, feature_indices(RECOVERY_FEATURES)]
        stress = x[:, feature_indices(STRESS_FEATURES)]
        muscle = x[:, feature_indices(MUSCLE_FEATURES)]
        metabolism = x[:, feature_indices(METABOLISM_FEATURES)]
        aging = x[:, feature_indices(AGING_FEATURES)]

        energy_mu, energy_log_var, energy_z = self.energy_encoder.forward(energy)

        recovery_mu, recovery_log_var, recovery_z = self.recovery_encoder.forward(recovery)

        stress_mu, stress_log_var, stress_z = self.stress_encoder.forward(stress)

        muscle_mu, muscle_log_var, muscle_z = self.muscle_encoder.forward(muscle)

        metabolism_mu, metabolism_log_var, metabolism_z = self.metabolism_encoder.forward(metabolism)

        aging_mu, aging_log_var, aging_z = self.aging_encoder.forward(aging)

        # ===================================================
        # Local Reconstructions
        # ===================================================

        energy_reconstruction = self.energy_decoder.forward(
            energy_z
        )

        recovery_reconstruction = self.recovery_decoder.forward(
            recovery_z
        )

        stress_reconstruction = self.stress_decoder.forward(
            stress_z
        )

        muscle_reconstruction = self.muscle_decoder.forward(
            muscle_z
        )

        metabolism_reconstruction = self.metabolism_decoder.forward(
            metabolism_z
        )

        aging_reconstruction = self.aging_decoder.forward(
            aging_z
        )
        return {

            "energy": energy_z,
            "recovery": recovery_z,
            "stress": stress_z,
            "muscle": muscle_z,
            "metabolism": metabolism_z,
            "aging": aging_z,

            # Local reconstructions

            "energy_reconstruction": energy_reconstruction,
            "recovery_reconstruction": recovery_reconstruction,
            "stress_reconstruction": stress_reconstruction,
            "muscle_reconstruction": muscle_reconstruction,
            "metabolism_reconstruction": metabolism_reconstruction,
            "aging_reconstruction": aging_reconstruction,

            "energy_mu": energy_mu,
            "recovery_mu": recovery_mu,
            "stress_mu": stress_mu,
            "muscle_mu": muscle_mu,
            "metabolism_mu": metabolism_mu,
            "aging_mu": aging_mu,

            "energy_log_var": energy_log_var,
            "recovery_log_var": recovery_log_var,
            "stress_log_var": stress_log_var,
            "muscle_log_var": muscle_log_var,
            "metabolism_log_var": metabolism_log_var,
            "aging_log_var": aging_log_var,


        }

    def backward_global(self, grads):
        self.energy_encoder.backward(grads["energy"])
        self.recovery_encoder.backward(grads["recovery"])
        self.stress_encoder.backward(grads["stress"])
        self.muscle_encoder.backward(grads["muscle"])
        self.metabolism_encoder.backward(grads["metabolism"])
        self.aging_encoder.backward(grads["aging"])

    def backward_local(self, grads):
        grad_energy = self.energy_decoder.backward(grads["energy"])
        self.energy_encoder.backward(grad_energy)

        grad_recovery = self.recovery_decoder.backward(grads["recovery"])
        self.recovery_encoder.backward(grad_recovery)

        grad_stress = self.stress_decoder.backward(grads["stress"])
        self.stress_encoder.backward(grad_stress)

        grad_muscle = self.muscle_decoder.backward(grads["muscle"])
        self.muscle_encoder.backward(grad_muscle)

        grad_metabolism = self.metabolism_decoder.backward(grads["metabolism"])
        self.metabolism_encoder.backward(grad_metabolism)

        grad_aging = self.aging_decoder.backward(grads["aging"])
        self.aging_encoder.backward(grad_aging)

    def update(self, lr):

        self.energy_encoder.update(lr)
        self.recovery_encoder.update(lr)
        self.stress_encoder.update(lr)
        self.muscle_encoder.update(lr)
        self.metabolism_encoder.update(lr)
        self.aging_encoder.update(lr)

        self.energy_decoder.update(lr)
        self.recovery_decoder.update(lr)
        self.stress_decoder.update(lr)
        self.muscle_decoder.update(lr)
        self.metabolism_decoder.update(lr)
        self.aging_decoder.update(lr)

    def zero_grad(self):

        self.energy_encoder.zero_grad()
        self.recovery_encoder.zero_grad()
        self.stress_encoder.zero_grad()
        self.muscle_encoder.zero_grad()
        self.metabolism_encoder.zero_grad()
        self.aging_encoder.zero_grad()

        self.energy_decoder.zero_grad()
        self.recovery_decoder.zero_grad()
        self.stress_decoder.zero_grad()
        self.muscle_decoder.zero_grad()
        self.metabolism_decoder.zero_grad()
        self.aging_decoder.zero_grad()

    def save(self):

        return {

            "energy_encoder": self.energy_encoder.save(),
            "recovery_encoder": self.recovery_encoder.save(),
            "stress_encoder": self.stress_encoder.save(),
            "muscle_encoder": self.muscle_encoder.save(),
            "metabolism_encoder": self.metabolism_encoder.save(),
            "aging_encoder": self.aging_encoder.save(),

            "energy_decoder": self.energy_decoder.save(),
            "recovery_decoder": self.recovery_decoder.save(),
            "stress_decoder": self.stress_decoder.save(),
            "muscle_decoder": self.muscle_decoder.save(),
            "metabolism_decoder": self.metabolism_decoder.save(),
            "aging_decoder": self.aging_decoder.save()

        }

    def load(self, state):

        self.energy_encoder.load(state["energy_encoder"])
        self.recovery_encoder.load(state["recovery_encoder"])
        self.stress_encoder.load(state["stress_encoder"])
        self.muscle_encoder.load(state["muscle_encoder"])
        self.metabolism_encoder.load(state["metabolism_encoder"])
        self.aging_encoder.load(state["aging_encoder"])

        self.energy_decoder.load(state["energy_decoder"])
        self.recovery_decoder.load(state["recovery_decoder"])
        self.stress_decoder.load(state["stress_decoder"])
        self.muscle_decoder.load(state["muscle_decoder"])
        self.metabolism_decoder.load(state["metabolism_decoder"])
        self.aging_decoder.load(state["aging_decoder"])