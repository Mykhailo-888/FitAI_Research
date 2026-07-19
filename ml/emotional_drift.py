import numpy as np


def analyze_emotional_drift(
    stress_level,
    alcohol_units=0,
    days=7,
    n_sim=10,
    seed=None,
    random_seed=None,
    rng=None,
    return_details=False,
):
    """
    Прогноз емоційного стресу на 7 днів за допомогою стохастичного рівняння Іто.

    Математична модель:
        dS_t = [κ (θ - S_t) + μ S_t + alcohol_drift] dt + σ_eff(S_t) dW_t

    Де:
        S_t          - рівень емоційного стресу в момент t
        θ = 5.5      - довгострокова рівновага (нормальний рівень стресу)
        κ            - швидкість повернення до норми (mean reversion)
        μ            - слабкий природний дрейф
        σ_eff        - ефективна волатильність (залежить від алкоголю)
        alcohol_drift - додатковий дрейф від алкоголю
        dW_t         - приріст Вінерівського процесу (випадковий шум)

    Алкоголь впливає нелінійно: чим більше алкоголю, тим сильніше зростає стрес і нестабільність.
    """
    # Парсинг вхідних даних
    try:
        S = float(stress_level)
    except (TypeError, ValueError):
        S = 5.0

    S = np.clip(S, 1.0, 10.0)
    alcohol = float(alcohol_units)

    if not isinstance(days, (int, np.integer)) or days <= 0:
        raise ValueError("days must be a positive integer")
    if not isinstance(n_sim, (int, np.integer)) or n_sim <= 0:
        raise ValueError("n_sim must be a positive integer")
    if seed is not None and random_seed is not None:
        raise ValueError("Pass either seed or random_seed, not both")
    if random_seed is not None:
        seed = random_seed

    if rng is not None and seed is not None:
        raise ValueError("Pass either seed or rng, not both")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if rng is None:
        rng = np.random.default_rng(seed) if seed is not None else None

    # Параметри моделі
    theta = 5.5
    kappa = 0.13
    mu    = 0.002
    sigma = 0.042

    # Нелінійний вплив алкоголю
    alcohol_norm = min(alcohol / 10.0, 2.5)
    alcohol_effect = 1 / (1 + np.exp(-(alcohol_norm - 1.2)))

    alcohol_drift = alcohol_effect * 0.95          # сильний вплив на рівень стресу
    sigma_eff     = sigma * (1 + 1.8 * alcohol_effect)   # алкоголь робить стрес більш нестабільним

    # Симуляція
    trajectories = []
    for _ in range(n_sim):
        current = S
        trajectory = []
        for _ in range(days):
            dW = (
                rng.normal(0, 1.0)
                if rng is not None
                else np.random.normal(0, 1.0)
            )
            drift = kappa * (theta - current) + mu * current + alcohol_drift
            diffusion = sigma_eff * current * dW
            current += drift + diffusion
            current = np.clip(current, 1.0, 10.0)
            trajectory.append(float(current))
        trajectories.append(trajectory)

    mean_trajectory = np.mean(np.asarray(trajectories, dtype=float), axis=0)
    final_stress = round(float(mean_trajectory[-1]), 1)

    if not return_details:
        return final_stress

    return {
        "trajectory": [round(float(value), 4) for value in mean_trajectory],
        "mean_stress": round(float(np.mean(mean_trajectory)), 4),
        "final_stress": round(float(mean_trajectory[-1]), 4),
        "volatility": round(float(np.std(mean_trajectory)), 4),
    }


# Тест
if __name__ == "__main__":
    print("Stress 5.0, alcohol 0  →", analyze_emotional_drift(5.0, 0))
    print("Stress 5.0, alcohol 5  →", analyze_emotional_drift(5.0, 5))
    print("Stress 5.0, alcohol 12 →", analyze_emotional_drift(5.0, 12))
    print("Stress 5.0, alcohol 25 →", analyze_emotional_drift(5.0, 25))
