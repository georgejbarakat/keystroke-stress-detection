# Predicting Cognitive Stress with Keystroke & Mouse Dynamics

**DSCI-441 — Statistical & Machine Learning · Final Project**
Graham Phillips · George Barakat · April 2026

A behavioral-biometrics pipeline that predicts whether a user is currently
experiencing self-reported cognitive stress, based on a 10-minute window of
their keystroke and mouse activity.

## Project description

We engineer 13 behavioral features (9 keystroke + 4 mouse) from raw
press/release timestamps and mouse events, label each window with the
nearest preceding stress self-report, and compare four classifiers under
two evaluation regimes:

- **Random 5-fold CV** — measures within-user discrimination.
- **Leave-one-user-out CV** — measures cross-user generalization.

The included Streamlit app captures live typing in the browser, computes
features client-side, and returns a stress probability from the trained
Random Forest model.

## Headline results (n = 464 windows, 2 users)

| Model              | Random 5-fold AUC | Leave-one-user-out AUC |
|--------------------|-------------------|------------------------|
| LogReg (M1 baseline) | 0.627 ± 0.062     | 0.530                  |
| **Random Forest** ⭐ | **0.704 ± 0.059** | 0.526                  |
| Gradient Boosting  | 0.651 ± 0.042     | 0.544                  |
| SVM-RBF            | 0.636 ± 0.025     | 0.537                  |

The 0.70 random-CV AUC is a meaningful improvement over the 0.603 baseline.
The collapse to ~0.53 under leave-one-user-out is the central finding of
this milestone: with only 2 labeled users, the model is mostly learning
individual typing habits, not a universal stress signature. Honest reporting
of this gap is more valuable than headline numbers from random CV.

## Data sources

1. **Stress Detection by Keystroke & Mouse** (Kaggle)
   <https://www.kaggle.com/datasets/anmolkumar/stress-detection-by-typing-pattern>
   — 2 users, 27,233 keystroke events, 5.2M mouse events, 98 self-reported
   stress/fatigue labels. Used for supervised training and evaluation.

2. **IKDD Keystroke Dynamics** (MDPI / GitHub)
   — 374 user sessions of raw inter-key intervals with demographic metadata.
   Used for distributional analysis and feature validation.

See `data/readme_data.txt` for download and placement instructions.

## Repository layout

```
.
├── data/
│   └── readme_data.txt          # how to obtain & place the raw data
├── features.py                  # feature engineering (loaders + sliding-window)
├── train_models.py              # main training script (run this end-to-end)
├── app.py                       # Streamlit web app
├── Milestone02_DSCI441.ipynb    # full analysis notebook
├── best_model.joblib            # trained model + metadata (created by train_models.py)
├── model_results.json           # CV results (created by train_models.py)
├── requirements.txt
└── ReadMe.txt                   # this file
```

## Required packages

```
python>=3.10
numpy
pandas
scikit-learn>=1.3
scipy
matplotlib
seaborn
streamlit
joblib
```

Install with `pip install -r requirements.txt`.

## How to run

```bash
# 1. Place data per data/readme_data.txt, then build features and train models:
python train_models.py
# This produces best_model.joblib and model_results.json.

# 2. Launch the live-typing demo:
streamlit run app.py
# The app loads best_model.joblib and accepts either live typing capture
# or manual feature entry.
```

The Jupyter notebook (`Milestone02_DSCI441.ipynb`) walks through the same
pipeline interactively, including hypothesis testing and visualization.

## Limitations

- **n = 2 users** for the supervised dataset. Leave-one-user-out CV is
  therefore 2-fold, and conclusions about cross-user generalization are
  preliminary.
- The labeling strategy (nearest preceding stress report within 30 minutes)
  treats stress as piecewise-constant between self-reports, which is a
  simplification.
- Feature extraction assumes continuous typing within a window; very sparse
  windows are dropped (`< 5 keys`).

## Future work

- Per-user feature normalization (z-score against each user's baseline) to
  separate cross-user stress signal from per-user typing fingerprint.
- Larger user pool — replicate the labeling protocol with ≥ 20 users so
  user-held-out evaluation has statistical power.
- Joint modeling with the IKDD dataset using its IKI distributions as a
  population-level prior.
