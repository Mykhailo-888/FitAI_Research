# FitAI — Physiologically Reactive AI Fitness System
## Live Demo
https://fitai-research.onrender.com/

>Press "Get Started" to explore the onboarding flow and interactive physiological AI dashboard.
FitAI is an experimental AI-driven physiological fitness platform built with Django, NumPy, OpenCV, and custom mathematical modeling.
<img width="1585" height="841" alt="image" src="https://github.com/user-attachments/assets/6f620f68-e6e5-4f84-bdd8-340d2900ce47" />



The project combines:

* 23 physiological input parameters
* OpenCV-based body photo analysis
* custom NumPy neural networks
* stochastic physiological simulation
* explainability analysis
* dynamic training optimization
* physiological risk modeling
* longitudinal monitoring systems
* temporal prediction tracking

Unlike traditional static fitness calculators, FitAI is designed as a physiologically reactive neural system rather than a fixed rule-based predictor.

## Physiologically Reactive Neural System
> The monitoring system tracks longitudinal neural predictions, physiological stability, and adaptive output dynamics over time.
<img width="1536" height="1024" alt="Physiological Responce Analysis" src="https://github.com/user-attachments/assets/c8813a25-260a-40d2-9bd8-a122beede09b" />

The system dynamically adapts predictions based on:
* HRV
* sleep quality
* emotional stress
* inflammation markers
* alcohol exposure
* cardiovascular state
* hormonal signals
* recovery dynamics

---

# Core Features

## Physiological Prediction Engine

FitAI predicts 8 physiological and fitness-related outputs:

* Calories Burned
* 1 km Run Time
* Cooper Test Distance
* Max Pull-Ups
* Burpees Capacity
* 10 km Run Time
* Waist Circumference Change
* Testosterone Projection

The prediction engine uses:
<img width="1536" height="1024" alt="Custom NumPy Neural Network" src="https://github.com/user-attachments/assets/33ce6fc1-5769-47b9-9df4-82c981cb0dc3" />

* custom NumPy neural networks
* hyperbolic tangent activations (`tanh`)
* softplus physiological output constraints
* momentum optimization
* gradient clipping
* early stopping
* normalization scaling
* physics-based target generation

The neural network is implemented entirely without high-level deep learning frameworks.

---

# Physiological Input System

The system combines:

* 23 physiological parameters
<img width="1536" height="1024" alt="Physiological Signal Flow" src="https://github.com/user-attachments/assets/e75d51c8-107c-46ff-a398-33da00805a1b" />


* OpenCV body proportion analysis from uploaded user photos

Physiological inputs include:

* HRV
* Sleep
* Emotional Stress
* CRP
* Testosterone
* Cortisol
* Resting Heart Rate
* Blood Pressure
* Waist Circumference
* Alcohol Exposure
* Recovery Metrics
* Performance Metrics

The OpenCV analysis module extracts:
<img width="1536" height="1024" alt="OpenCV Body Analysis" src="https://github.com/user-attachments/assets/189e3310-8ca8-43a5-b7cb-1bc948184250" />

* shoulder width
* waist width
* hip width
* body ratios
* physique classification
* training recommendations

---

# Physiologically Reactive Neural Architecture

A major design goal of FitAI was to avoid creating a random black-box predictor.

The neural system was explicitly designed to demonstrate:

* structured physiological behavior
* parameter sensitivity
* explainability
* stable physiological responses

The model behavior is validated through:

## Relative Perturbation Analysis (+30%)

Each physiological feature is independently perturbed by +30% to measure how predictions react.

## Gradient-Based Sensitivity Analysis (∂y/∂x)
<img width="685" height="572" alt="image" src="https://github.com/user-attachments/assets/07dbb93e-560a-483e-b0d6-845e5dc8dcb1" />

Local physiological sensitivity is estimated numerically through gradient-based feature analysis.

This means the model does not simply output arbitrary values.

## The network dynamically and consistently reacts to physiological changes.

Predictions structurally respond to:
* HRV
* sleep quality
* emotional stress
* CRP inflammation markers
* alcohol exposure
* testosterone
* cardiovascular parameters

Observed physiological behaviors include:

* increased sleep improving recovery-related outputs
* emotional stress worsening endurance predictions
* elevated CRP reducing aerobic performance
* HRV strongly affecting endurance and recovery metrics
* alcohol negatively affecting recovery and hormonal projections

FitAI therefore behaves as a:

## Structured Physiological Response Model

rather than a purely statistical estimator.

---

# Explainability System

FitAI includes a built-in explainability pipeline.

The project generates:

* `feature_importance.json`
* per-output sensitivity analysis
* relative perturbation analysis
* gradient sensitivity analysis

This allows visualization of:

* which physiological parameters affect each output
* how strongly the neural system reacts
* whether predictions remain physiologically consistent

Examples:

* HRV strongly affects Cooper performance
* Sleep strongly affects Testosterone and recovery
* Emotional stress strongly affects endurance metrics
* CRP affects inflammation-sensitive outputs

---

# Model Evaluation & Monitoring System

FitAI includes an experimental physiological monitoring and evaluation framework designed to track neural network behavior over time.

The platform does not simply display predictions.

Instead, it continuously monitors:

* prediction consistency
* physiological trend dynamics
* temporal stability
* recovery behavior
* cardiovascular response
* hormonal tendencies
* prediction drift across sessions

The evaluation architecture combines two independent systems.

---

## Stability Metrics

The Stability system measures real neural network behavior across historical onboarding sessions.

Tracked statistics include:

* Stability Mean
* Stability Standard Deviation
* Stability Drift

This layer evaluates:

* prediction consistency
* longitudinal behavior
* temporal variance
* real historical neural response dynamics

Stability metrics are calculated directly from historical prediction records generated by the neural network itself.

---

## Synthetic Metrics
<img width="985" height="749" alt="s2" src="https://github.com/user-attachments/assets/9af2cdde-a38c-44c6-855b-9a7a45070e65" />

The Synthetic system is a separate rule-based physiological simulation layer.

It generates lightweight physiological reference estimates using:

* HRV
* sleep
* emotional stress
* CRP
* weight
* cardiovascular signals

Synthetic metrics include:

* Synthetic Mean
* Synthetic Standard Deviation
* Synthetic Drift

This system acts as:

* a physiological comparison baseline
* a simulation-oriented monitoring layer
* an auxiliary consistency reference

The synthetic layer is implemented independently from the neural network through the `synthetic_ground_truth()` simulation pipeline.

---

## Dual-Layer Evaluation Architecture
<img width="1536" height="1024" alt="Dual-Layer Evaluation Architecture" src="https://github.com/user-attachments/assets/8e289796-b908-4999-a9d1-f0d1d60526c5" />

FitAI therefore implements a:

# Dual-Layer Model Evaluation System

combining:

* real neural prediction stability
* synthetic physiological simulation comparison

This architecture allows the platform to monitor:

* model drift
* physiological consistency
* behavioral stability
* abnormal prediction deviations
* recovery trend evolution

---

# Model Health Score

FitAI computes an experimental:

## Model Health Score

derived from:

* stability drift
* synthetic drift
* historical prediction variance

The score acts as a lightweight behavioral monitoring signal for neural prediction consistency.

The purpose of the system is not medical validation, but:

* neural behavior monitoring
* physiological trend visualization
* experimental AI evaluation
* recovery-oriented tracking
---

# Dynamic Visualization System

FitAI includes dynamic physiological dashboards built with:

* Django
* JavaScript
* Chart.js

The visualization layer provides:

* prediction timelines
* historical physiological graphs
* model stability charts
* explainability displays
* monitoring dashboards
* temporal drift visualization
* <img width="1579" height="843" alt="s1" src="https://github.com/user-attachments/assets/364c761e-a30a-4909-90d0-bca549bc76ce" />

---
# Emotional Drift Modeling
<img width="1486" height="962" alt="stochastic" src="https://github.com/user-attachments/assets/557a7741-bbbb-40b1-98fe-7baffd309849" />

FitAI contains a stochastic emotional forecasting system inspired by Itô stochastic differential equations.

Mathematical form:

```math
dS_t = [\kappa(\theta - S_t) + \mu S_t + alcohol_{drift}]dt + \sigma_{eff}(S_t)dW_t
```

The system models:

* emotional stress drift
* nonlinear alcohol influence
* volatility growth
* stochastic instability
* mean reversion dynamics

Alcohol exposure increases:

* stress drift
* emotional instability
* volatility of future emotional states

This module introduces stochastic physiological modeling into the platform.

---

# Training Optimization System
<img width="1536" height="1024" alt="Physiological Signal Flow" src="https://github.com/user-attachments/assets/b7210051-d12e-4913-8a73-aea2fd4a2a04" />

FitAI includes a physiological weekly training optimizer.

The optimizer dynamically adjusts:

* calorie deficit
* weekly HIIT count
* recovery load
* cardiovascular stress
* recovery penalties

The optimizer considers:

* HRV
* sleep quality
* alcohol exposure
* blood pressure
* age
* training load
* recovery penalties

The system automatically reduces training intensity when recovery risk becomes elevated.
---
# Hamilton–Jacobi–Bellman Inspired Risk Modeling

>FitAI includes a simplified physiological control model inspired by Hamilton–Jacobi–Bellman optimization ideas.
<img width="1536" height="1024" alt="hjb" src="https://github.com/user-attachments/assets/8e2b4d91-88ca-401a-bce8-a829d9c18b4c" />



The system estimates:

* training risk
* optimal training intensity
* short-term physiological trajectory

The model simulates:

* HRV dynamics
* sleep recovery dynamics
* blood pressure response
* physiological cost minimization

Simplified objective form:

```math
J(X,u) = \sum instant\_cost(X_t,u_t)
```
where:

* (X_t) → physiological state
* (u_t) → training intensity/control signal

The system searches for lower-risk physiological trajectories across time.

This module acts as a simplified optimal-control inspired recovery system.
---
# Dataset Architecture

FitAI uses a dual-dataset architecture.

## Base Physiological Dataset

`edited_23_params_realistic.csv`

Contains:

* engineered physiological variables
* observable biomedical signals
* synthetic physiological feature generation
* stable production-oriented training data

Generated using:

* pandas
* NumPy
* physiological feature engineering
---
## Latent Physiological Dataset
<img width="1536" height="1024" alt="Latent Physiological States" src="https://github.com/user-attachments/assets/1a0af026-8a26-4672-b990-cf20f3c1f260" />

`edited_23_params_realistic_latent.csv`

Experimental hidden-state physiological dataset containing:

* latent_energy
* latent_stress
* latent_recovery
* latent_rhythm

The latent pipeline introduces:

* hidden-state physiological representations
* normalized latent synthesis
* experimental recovery modeling
* future research-oriented architecture

FitAI therefore supports architectural switching between:

* observable physiological learning
* latent hidden-state augmented learning

The latent dataset pipeline is reserved for future experimental research and advanced model extensions.
---
# Neural Network Architecture

Main architecture:

* Input: 23 physiological parameters
* Embedding layer: 48 neurons
* Hidden layer: 32 neurons
* Output layer: 8 physiological outputs

Activation functions:

* Hidden layers → tanh
* Output constraints → softplus + tanh

Additional features:

* momentum optimization
* gradient clipping
* early stopping
* normalization scaling
* physiological output constraints
---

# Model Serialization

The trained model is stored as:

`trained_fitness_model_simple.pkl`

Serialized components include:

* neural network weights
* embeddings
* normalization statistics
* output scaling state
* validation metrics

The utility script:

`check_model.py`

converts serialized model parameters into a human-readable diagnostic overview.
---
# Testing

The project includes:

* physiological prediction tests
* behavioral consistency tests
* deterministic inference tests
* validation tests
* Django form validation tests

Implemented using:

* pytest
---
# Technology Stack

## Backend

* Python
* Django
* NumPy
* pandas
* OpenCV
* SQLite

## Machine Learning
<img width="1536" height="1024" alt="Custom NumPy Neural Network" src="https://github.com/user-attachments/assets/2d249e0e-835f-4edb-a1fc-62312fbbf0bc" />

* custom NumPy neural networks
* explainability analysis
* stochastic physiological simulation
* sensitivity analysis
* physiological monitoring systems

## Infrastructure

* Docker
* docker-compose
* pytest

## Frontend

* HTML
* CSS
* JavaScript
* Chart.js
---
# Project Structure

```text
FitAI/
│
├── data/
│   ├── edited_23_params_realistic.csv
│   │   # Expanded physiological dataset generated with pandas feature engineering
│   │
│   ├── edited_23_params_realistic_latent.csv
│   │   # Physiological training dataset augmented with hidden-state latent features
│   │
│   └── gym_members_exercise_tracking.csv
│       # Original dataset downloaded from Kaggle
│
├── fitai/
│   ├── settings.py
│   └── urls.py
│
├── fitness/
│   ├── migrations/
│   │
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   │
│   ├── templates/
│   │   ├── base.html
│   │   ├── history.html
│   │   ├── metrics.html
│   │   ├── onboarding.html
│   │   ├── results.html
│   │   └── update.html
│   │
│   ├── apps.py
│   ├── forms.py
│   ├── urls.py
│   │
│   └── views.py
│       # Core Django application logic integrating onboarding,
│       # physiological prediction, explainability, training optimization,
│       # risk analysis, photo analysis, and model stability metrics
│
├── media/
│   └── photos/
│
├── ml/
│   ├── models/
│   │   ├── check_model.py
│   │   │   # Utility script for converting serialized .pkl model contents
│   │   │   # into a human-readable parameter overview
│   │   │
│   │   └── trained_fitness_model_simple.pkl
│   │       # Serialized trained physiological prediction model
│   │       # with weights and normalization state
│   │
│   ├── emotional_drift.py
│   │   # Stochastic emotional stress forecasting using an
│   │   # Itô-process-inspired physiological drift model
│   │
│   ├── feature_importance.json
│   │   # Feature importance and explainability analysis for model outputs
│   │
│   ├── fit_model_core.py
│   │   # Custom NumPy neural network with physiological output constraints,
│   │   # momentum optimization, early stopping, and feature importance analysis
│   │
│   ├── photo_analysis.py
│   │   # OpenCV-based body proportion analysis and physique classification
│   │   # from user photos
│   │
│   ├── preprocess_dataset.py
│   │   # Physiological dataset expansion and synthetic feature engineering
│   │   # pipeline using pandas and NumPy
│   │
│   ├── preprocess_latent.py
│   │   # Latent physiological feature generation and hidden-state
│   │   # normalization pipeline
│   │
│   ├── train_model.py
│   │   # NumPy neural network training pipeline with physics-based
│   │   # target generation, validation, serialization,
│   │   # and explainability analysis
│   │
│   ├── training_history.json
│   │   # Training configuration, validation metrics, and model metadata
│   │
│   ├── training_optimizer.py
│   │   # Physiological weekly training load optimizer with recovery
│   │   # and cardiovascular risk adjustment
│   │
│   └── training_risk.py
│       # Simplified Hamilton–Jacobi–Bellman-inspired physiological
│       # training risk and intensity optimization model
│
├── tests/
│   ├── test_model.py
│   │   # Pytest-based validation suite for physiological prediction
│   │   # consistency and behavioral sanity checks
│   │
│   └── test_views.py
│       # Input validation tests for Django form and physiological data handling
│
├── .dockerignore
├── .gitignore
├── db.sqlite3
├── docker-compose.yml
├── Dockerfile
├── main.py
├── manage.py
├── pytest.ini
└── requirements.txt
```
---
# Research Direction

Future directions include:

* latent-state physiological learning
* hidden recovery state modeling
* nonlinear regression extensions
* sinusoidal physiological dynamics
* quaternion-based neural layers
* physiological hidden-state embeddings
* advanced optimal-control systems
* online adaptive learning
* stochastic recovery simulation

---

# FitAI Philosophy

FitAI was designed not as a generic calorie predictor, but as an experimental physiologically reactive AI system.

The project attempts to combine:

* machine learning
* physiological modeling
* explainability
* stochastic mathematics
* optimal control ideas
* recovery dynamics
* hidden-state modeling
* longitudinal monitoring
* physiological simulation

into a single extensible research-oriented architecture.
