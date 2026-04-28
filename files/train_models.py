"""
M2 modeling: random and user-held-out evaluation across LR, RF, GBT, SVM.
Hyperparameter tuning is done via grid search with stratified CV.
The best model is refit on all data and saved for the Streamlit app.

Run end-to-end:
    python train_models.py
This will build the feature dataset, evaluate all models, and save
best_model.joblib + model_results.json + feature_dataset.csv.
"""
import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, GridSearchCV, cross_val_score
)
from sklearn.metrics import roc_auc_score

import features as F


# ───────────────────────── Config ───────────────────────────
KAGGLE_PATH = os.environ.get("KAGGLE_PATH", "./data/KeyStroke")
USERS       = ["user_1", "user_2"]


# ─────────────────── Build / load feature dataset ────────────
FEATS_CSV = "feature_dataset.csv"
if os.path.exists(FEATS_CSV):
    print(f"Loading cached features from {FEATS_CSV}")
    DF = pd.read_csv(FEATS_CSV)
else:
    print("Building feature dataset from raw data...")
    ks, uc, md = F.load_all(KAGGLE_PATH, USERS)
    uc = F.add_labels(uc)
    DF = F.build_feature_dataset(ks, uc, md, window_minutes=10, overlap=0.5)
    DF.to_csv(FEATS_CSV, index=False)
    print(f"Saved {FEATS_CSV}")

DF = DF.dropna(subset=['stress_label'])
print(f"Modeling on {len(DF)} windows")
print(f"Class balance: {DF['stress_label'].value_counts().to_dict()}")
print(f"Per user: {DF.groupby('user').size().to_dict()}")

X      = DF[F.ALL_FEATURES].values
y      = DF['stress_label'].astype(int).values
groups = DF['user'].values


def make_pipeline(estimator):
    """Standard preprocessing + estimator. Imputer handles any NaN mouse cells."""
    return Pipeline([
        ("imp",    SimpleImputer(strategy='median')),
        ("scaler", StandardScaler()),
        ("clf",    estimator),
    ])


# ───────────────── Models + hyperparameter grids ─────────────
MODELS = {
    "LogReg": (
        make_pipeline(LogisticRegression(max_iter=2000, class_weight='balanced',
                                          random_state=42)),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "RandomForest": (
        make_pipeline(RandomForestClassifier(class_weight='balanced',
                                             random_state=42, n_jobs=1)),
        {
            "clf__n_estimators": [200],
            "clf__max_depth":    [None, 8],
        },
    ),
    "GradBoost": (
        make_pipeline(GradientBoostingClassifier(random_state=42)),
        {
            "clf__n_estimators":  [100, 200],
            "clf__max_depth":     [2, 3],
            "clf__learning_rate": [0.1],
        },
    ),
    "SVM-RBF": (
        make_pipeline(SVC(kernel='rbf', probability=True,
                          class_weight='balanced', random_state=42)),
        {"clf__C": [1.0, 5.0], "clf__gamma": ['scale']},
    ),
}


def bootstrap_ci(scores, n_boot=2000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    boot = [np.mean(rng.choice(scores, size=len(scores), replace=True))
            for _ in range(n_boot)]
    a = (1 - ci) / 2
    return np.quantile(boot, a), np.quantile(boot, 1 - a)


# ─────────────────── Evaluation A: random 5-fold CV ──────────
print("\n" + "=" * 64)
print("A) RANDOM 5-FOLD CV  (overestimates generalization)")
print("=" * 64)
random_results = {}
for name, (pipe, grid) in MODELS.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    gs = GridSearchCV(pipe, grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
    scores = cross_val_score(gs, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    lo, hi = bootstrap_ci(scores)
    random_results[name] = {
        "auc_mean": float(scores.mean()),
        "auc_std":  float(scores.std()),
        "ci_lo":    float(lo),
        "ci_hi":    float(hi),
        "scores":   [float(s) for s in scores],
    }
    print(f"  {name:14s}  AUC = {scores.mean():.3f} ± {scores.std():.3f}   "
          f"95% CI [{lo:.3f}, {hi:.3f}]")


# ─────────────── Evaluation B: leave-one-USER-out ────────────
print("\n" + "=" * 64)
print("B) LEAVE-ONE-USER-OUT CV  (true generalization across users)")
print("=" * 64)
group_results = {}
gkf = GroupKFold(n_splits=len(np.unique(groups)))
for name, (pipe, grid) in MODELS.items():
    fold_aucs = []
    for tr_idx, te_idx in gkf.split(X, y, groups):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        # Inner CV uses GroupKFold too -- but with only 2 users we tune on a
        # stratified split of the training fold instead.
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        gs = GridSearchCV(pipe, grid, cv=inner, scoring='roc_auc', n_jobs=-1)
        gs.fit(Xtr, ytr)
        prob = gs.predict_proba(Xte)[:, 1]
        # AUC undefined if a fold's test set has only one class -- guard
        if len(np.unique(yte)) > 1:
            fold_aucs.append(roc_auc_score(yte, prob))
    fold_aucs = np.array(fold_aucs)
    group_results[name] = {
        "auc_mean":  float(fold_aucs.mean()) if len(fold_aucs) else float('nan'),
        "auc_per_fold": [float(s) for s in fold_aucs],
        "n_folds":   len(fold_aucs),
    }
    print(f"  {name:14s}  AUC = {fold_aucs.mean():.3f} "
          f"(folds: {fold_aucs.round(3).tolist()})")


# ─────────────────── Pick winner & refit on all data ──────────
print("\n" + "=" * 64)
print("C) FINAL MODEL (refit on full dataset)")
print("=" * 64)
# Pick by random-CV AUC -- with only 2 users LOUO is too noisy to be the sole
# selection criterion (one fold = one user).
best_name = max(random_results, key=lambda k: random_results[k]["auc_mean"])
print(f"Best model on random CV: {best_name}  "
      f"(AUC {random_results[best_name]['auc_mean']:.3f})")

pipe, grid = MODELS[best_name]
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_gs = GridSearchCV(pipe, grid, cv=inner, scoring='roc_auc', n_jobs=-1)
final_gs.fit(X, y)
print(f"Best hyperparameters: {final_gs.best_params_}")
print(f"Best CV AUC: {final_gs.best_score_:.3f}")

# Save artifacts for the Streamlit app
joblib.dump({
    "model":    final_gs.best_estimator_,
    "features": F.ALL_FEATURES,
    "model_name": best_name,
    "best_params": final_gs.best_params_,
    "random_cv_auc": random_results[best_name]["auc_mean"],
    "louo_cv_auc":   group_results[best_name]["auc_mean"],
}, 'best_model.joblib')

# Save results JSON for the report / poster
with open('model_results.json', 'w') as f:
    json.dump({
        "random_cv":      random_results,
        "leave_one_user_out": group_results,
        "best_model":     best_name,
        "best_params":    final_gs.best_params_,
        "n_samples":      int(len(X)),
        "class_balance":  {str(k): int(v) for k, v in
                           pd.Series(y).value_counts().to_dict().items()},
    }, f, indent=2)
print("\nSaved best_model.joblib and model_results.json")
