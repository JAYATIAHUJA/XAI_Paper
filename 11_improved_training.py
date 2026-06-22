"""
=============================================================
STEP 11 — Improved Training Pipeline
=============================================================
Changes from baseline:
  1. 5-Fold Cross Validation (not single split)
  2. RandomizedSearchCV for XGBoost hyperparameter tuning
  3. No class_weight='balanced' on LR (removes SPD anomaly)
  4. CFVR-Prob (existing) + CFVR-Class (new — counts prediction flips)
  5. All results reported with Mean ± Std

Output:
  data/improved_models.pkl
  outputs/cv_results.csv
  outputs/improved_results_table.csv
  outputs/cv_comparison.png
=============================================================
"""

import os, pickle, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     RandomizedSearchCV, train_test_split)
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (accuracy_score, f1_score, roc_auc_score,
                                    confusion_matrix, make_scorer)
from sklearn.preprocessing  import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose        import ColumnTransformer
from xgboost import XGBClassifier
from fairlearn.metrics import (demographic_parity_difference,
                                demographic_parity_ratio,
                                equalized_odds_difference)

print("\n" + "=" * 65)
print("  STEP 11 — IMPROVED TRAINING (5-Fold CV + Hyperparameter Tuning)")
print("=" * 65)

# ── Load processed data ───────────────────────────────────
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]
gender_test  = data["gender_test"]
gender_train = data["gender_train"]
feature_names = data["feature_names"]

# Gender index (for CFVR)
gender_idx = next(i for i, n in enumerate(feature_names) if "Gender_enc" in n)
print(f"\n[INFO] Train: {X_train.shape} | Test: {X_test.shape}")
print(f"[INFO] Gender feature index: {gender_idx} → '{feature_names[gender_idx]}'")

# ─────────────────────────────────────────────────────────
# CFVR-Prob: probability shift > threshold (existing)
# CFVR-Class: binary prediction flip (new metric)
# ─────────────────────────────────────────────────────────
def compute_cfvr_both(model, X_test, gender_idx, threshold=0.10):
    """Returns both CFVR-Prob and CFVR-Class."""
    p_orig = model.predict_proba(X_test)[:, 1]
    X_cf = X_test.copy()
    uniq = np.unique(X_cf[:, gender_idx])
    if len(uniq) == 2:
        a, b = uniq[0], uniq[1]
        ma, mb = X_cf[:, gender_idx] == a, X_cf[:, gender_idx] == b
        X_cf[ma, gender_idx] = b
        X_cf[mb, gender_idx] = a
    else:
        X_cf[:, gender_idx] = -X_cf[:, gender_idx]

    p_cf = model.predict_proba(X_cf)[:, 1]
    delta = np.abs(p_cf - p_orig)

    cfvr_prob  = (delta > threshold).mean()

    # CFVR-Class: did the binary prediction actually flip?
    pred_orig = (p_orig >= 0.5).astype(int)
    pred_cf   = (p_cf   >= 0.5).astype(int)
    cfvr_class = (pred_orig != pred_cf).mean()

    return round(cfvr_prob, 4), round(cfvr_class, 4)

# ─────────────────────────────────────────────────────────
# PART 1: 5-Fold Cross Validation (no tuning yet)
# ─────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  PART 1: 5-FOLD CROSS VALIDATION (All Models)")
print("-" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

base_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
        # NO class_weight='balanced' — removes artificial SPD anomaly
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "XGBoost (base)": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, eval_metric="logloss", verbosity=0
    ),
}

scoring = {
    "accuracy" : "accuracy",
    "f1"       : "f1",
    "roc_auc"  : "roc_auc",
}

cv_results = {}
for name, model in base_models.items():
    print(f"\n  [{name}] Running 5-fold CV...", end=" ", flush=True)
    t0 = time.time()
    cv_res = cross_validate(model, X_train, y_train,
                            cv=cv, scoring=scoring,
                            return_train_score=False, n_jobs=-1)
    elapsed = time.time() - t0

    acc_mean  = cv_res["test_accuracy"].mean()
    acc_std   = cv_res["test_accuracy"].std()
    f1_mean   = cv_res["test_f1"].mean()
    f1_std    = cv_res["test_f1"].std()
    auc_mean  = cv_res["test_roc_auc"].mean()
    auc_std   = cv_res["test_roc_auc"].std()

    cv_results[name] = {
        "Accuracy"     : acc_mean,  "Accuracy_std"  : acc_std,
        "F1"           : f1_mean,   "F1_std"        : f1_std,
        "ROC-AUC"      : auc_mean,  "ROC-AUC_std"   : auc_std,
    }
    print(f"done ({elapsed:.1f}s)")
    print(f"    Accuracy : {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"    F1       : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"    ROC-AUC  : {auc_mean:.4f} ± {auc_std:.4f}")

# ─────────────────────────────────────────────────────────
# PART 2: XGBoost Hyperparameter Tuning
# ─────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  PART 2: XGBOOST HYPERPARAMETER TUNING (RandomizedSearchCV)")
print("  Searching over: max_depth, learning_rate, n_estimators,")
print("                  subsample, colsample_bytree, min_child_weight")
print("-" * 65)

param_dist = {
    "n_estimators"     : [100, 200, 300, 400, 500],
    "max_depth"        : [3, 4, 5, 6, 7, 8],
    "learning_rate"    : [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    "subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree" : [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight" : [1, 3, 5, 7],
    "gamma"            : [0, 0.1, 0.2, 0.3],
    "reg_alpha"        : [0, 0.01, 0.1, 0.5],
    "reg_lambda"       : [0.5, 1.0, 1.5, 2.0],
}

xgb_base = XGBClassifier(
    random_state=42, eval_metric="logloss", verbosity=0
)

rscv = RandomizedSearchCV(
    xgb_base, param_dist,
    n_iter=50,             # 50 random combinations
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
    verbose=1,
    refit=True
)

print("\n  Fitting 50 candidates × 5 folds = 250 fits...")
t0 = time.time()
rscv.fit(X_train, y_train)
elapsed = time.time() - t0

print(f"\n  Done in {elapsed:.1f}s")
print(f"  Best ROC-AUC (CV): {rscv.best_score_:.4f}")
print(f"  Best params:")
for k, v in rscv.best_params_.items():
    print(f"    {k}: {v}")

best_xgb = rscv.best_estimator_

# 5-fold CV on best XGB
cv_best = cross_validate(best_xgb, X_train, y_train,
                          cv=cv, scoring=scoring, n_jobs=-1)
cv_results["XGBoost (tuned)"] = {
    "Accuracy"    : cv_best["test_accuracy"].mean(),
    "Accuracy_std": cv_best["test_accuracy"].std(),
    "F1"          : cv_best["test_f1"].mean(),
    "F1_std"      : cv_best["test_f1"].std(),
    "ROC-AUC"     : cv_best["test_roc_auc"].mean(),
    "ROC-AUC_std" : cv_best["test_roc_auc"].std(),
}
print(f"\n  Tuned XGB CV results:")
print(f"    Accuracy : {cv_results['XGBoost (tuned)']['Accuracy']:.4f} ± {cv_results['XGBoost (tuned)']['Accuracy_std']:.4f}")
print(f"    F1       : {cv_results['XGBoost (tuned)']['F1']:.4f} ± {cv_results['XGBoost (tuned)']['F1_std']:.4f}")
print(f"    ROC-AUC  : {cv_results['XGBoost (tuned)']['ROC-AUC']:.4f} ± {cv_results['XGBoost (tuned)']['ROC-AUC_std']:.4f}")

# ─────────────────────────────────────────────────────────
# PART 3: Final evaluation on held-out test set
# ─────────────────────────────────────────────────────────
print("\n" + "-" * 65)
print("  PART 3: FINAL EVALUATION ON HELD-OUT TEST SET")
print("  (Fairness metrics + CFVR-Prob + CFVR-Class)")
print("-" * 65)

# Refit all base models on full training set
final_models = {}
for name, model in base_models.items():
    model.fit(X_train, y_train)
    final_models[name] = model
final_models["XGBoost (tuned)"] = best_xgb
# best_xgb already fitted via refit=True in RSCV

final_results = {}
predictions_new = {}

for name, model in final_models.items():
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)

    spd = abs(demographic_parity_difference(y_test, y_pred, sensitive_features=gender_test))
    try:
        dir_ = demographic_parity_ratio(y_test, y_pred, sensitive_features=gender_test)
    except Exception:
        dir_ = float("nan")
    eod  = equalized_odds_difference(y_test, y_pred, sensitive_features=gender_test)

    cfvr_prob, cfvr_class = compute_cfvr_both(model, X_test, gender_idx)

    # Gender-wise selection rates
    f_mask = gender_test == 0
    m_mask = gender_test == 1
    f_rate = y_pred[f_mask].mean() * 100
    m_rate = y_pred[m_mask].mean() * 100

    final_results[name] = {
        "Accuracy"   : round(acc,  4),
        "F1"         : round(f1,   4),
        "ROC-AUC"    : round(auc,  4),
        "DIR"        : round(dir_, 4),
        "SPD"        : round(spd,  4),
        "EOD"        : round(eod,  4),
        "CFVR-Prob"  : cfvr_prob,
        "CFVR-Class" : cfvr_class,
        "Female%"    : round(f_rate, 1),
        "Male%"      : round(m_rate, 1),
    }
    predictions_new[name] = {"y_pred": y_pred, "y_proba": y_proba}
    print(f"\n  [{name}]")
    print(f"    Accuracy: {acc:.4f}  F1: {f1:.4f}  ROC-AUC: {auc:.4f}")
    print(f"    DIR: {dir_:.4f}  SPD: {spd:.4f}  EOD: {eod:.4f}")
    print(f"    CFVR-Prob: {cfvr_prob:.4f}  CFVR-Class: {cfvr_class:.4f}")
    print(f"    Female pred rate: {f_rate:.1f}%  |  Male pred rate: {m_rate:.1f}%")

# ─────────────────────────────────────────────────────────
# PART 4: Summary table
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("  FINAL RESULTS TABLE (With CFVR-Class added)")
print("=" * 100)
cols = ["Accuracy", "F1", "ROC-AUC", "DIR", "SPD", "EOD", "CFVR-Prob", "CFVR-Class"]
header = f"  {'Model':<26}" + "".join(f"{c:>13}" for c in cols)
print(header)
print("  " + "-" * 98)
for mname, metrics in final_results.items():
    row = f"  {mname:<26}" + "".join(f"{metrics[c]:>13.4f}" for c in cols)
    print(row)

print("\n  KEY: CFVR-Prob = |Δp| > 0.10 | CFVR-Class = prediction label flipped")
print("  DIR ideal=1.0 | SPD/EOD/CFVR ideal=0.0")

# ─────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────
# CV results CSV
cv_df = pd.DataFrame(cv_results).T[["Accuracy","Accuracy_std","F1","F1_std","ROC-AUC","ROC-AUC_std"]]
cv_df.index.name = "Model"
cv_df.to_csv("outputs/cv_results.csv")
print("\n[SAVED] outputs/cv_results.csv")

# Final results CSV
final_df = pd.DataFrame(final_results).T
final_df.index.name = "Model"
final_df.to_csv("outputs/improved_results_table.csv")
print("[SAVED] outputs/improved_results_table.csv")

# Save models
improved_bundle = {
    "final_models"   : final_models,
    "predictions"    : predictions_new,
    "final_results"  : final_results,
    "cv_results"     : cv_results,
    "best_xgb_params": rscv.best_params_,
    "best_xgb_cv_auc": rscv.best_score_,
}
with open("data/improved_models.pkl", "wb") as f:
    pickle.dump(improved_bundle, f)
print("[SAVED] data/improved_models.pkl")

# ─────────────────────────────────────────────────────────
# PART 5: Visualizations
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Improved Model Results: 5-Fold CV + Tuned XGBoost",
             fontsize=14, fontweight="bold")

model_names = list(cv_results.keys())
x = np.arange(len(model_names))
colors = ["#4C72B0", "#55A868", "#C44E52", "#9467BD"]

# 1. ROC-AUC with error bars (CV)
auc_means = [cv_results[m]["ROC-AUC"] for m in model_names]
auc_stds  = [cv_results[m]["ROC-AUC_std"] for m in model_names]
bars = axes[0, 0].bar(x, auc_means, yerr=auc_stds, capsize=6,
                      color=colors[:len(model_names)], alpha=0.85, edgecolor="white")
axes[0, 0].axhline(0.5, color="red", linestyle="--", linewidth=1.5, label="Random baseline")
axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(model_names, rotation=12, ha="right")
axes[0, 0].set_ylim(0.4, 0.75); axes[0, 0].set_ylabel("ROC-AUC")
axes[0, 0].set_title("5-Fold CV: ROC-AUC (Mean ± Std)")
axes[0, 0].legend()
for bar, m, s in zip(bars, auc_means, auc_stds):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.005,
                    f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

# 2. Accuracy with error bars (CV)
acc_means = [cv_results[m]["Accuracy"] for m in model_names]
acc_stds  = [cv_results[m]["Accuracy_std"] for m in model_names]
bars2 = axes[0, 1].bar(x, acc_means, yerr=acc_stds, capsize=6,
                        color=colors[:len(model_names)], alpha=0.85, edgecolor="white")
axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(model_names, rotation=12, ha="right")
axes[0, 1].set_ylim(0.4, 0.75); axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].set_title("5-Fold CV: Accuracy (Mean ± Std)")
for bar, m, s in zip(bars2, acc_means, acc_stds):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.005,
                    f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

# 3. Fairness metrics comparison
final_names = list(final_results.keys())
x2 = np.arange(len(final_names))
fair_metrics = ["SPD", "EOD", "CFVR-Prob", "CFVR-Class"]
bar_w = 0.2
for j, metric in enumerate(fair_metrics):
    vals = [final_results[m][metric] for m in final_names]
    axes[1, 0].bar(x2 + j * bar_w, vals, bar_w, label=metric,
                   alpha=0.85, edgecolor="white")
axes[1, 0].set_xticks(x2 + 1.5*bar_w)
axes[1, 0].set_xticklabels(final_names, rotation=12, ha="right")
axes[1, 0].axhline(0.1, color="red", linestyle="--", linewidth=1, label="Threshold 0.10")
axes[1, 0].set_ylabel("Score"); axes[1, 0].set_title("Fairness Metrics by Model")
axes[1, 0].legend(fontsize=8)

# 4. CFVR-Prob vs CFVR-Class comparison
cfvr_prob_vals  = [final_results[m]["CFVR-Prob"]  for m in final_names]
cfvr_class_vals = [final_results[m]["CFVR-Class"] for m in final_names]
axes[1, 1].scatter(cfvr_prob_vals, cfvr_class_vals,
                   color=colors[:len(final_names)], s=150, zorder=5)
for i, name in enumerate(final_names):
    axes[1, 1].annotate(name, (cfvr_prob_vals[i], cfvr_class_vals[i]),
                         textcoords="offset points", xytext=(8, 4), fontsize=9)
axes[1, 1].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4)
axes[1, 1].set_xlabel("CFVR-Prob (|Δp| > 0.10)")
axes[1, 1].set_ylabel("CFVR-Class (prediction flipped)")
axes[1, 1].set_title("CFVR-Prob vs CFVR-Class")
axes[1, 1].set_xlim(-0.05, 0.8); axes[1, 1].set_ylim(-0.05, 0.8)

plt.tight_layout()
plt.savefig("outputs/cv_comparison.png", dpi=150)
plt.close()
print("[SAVED] outputs/cv_comparison.png")

print("\n" + "=" * 65)
print("  COMPLETE — Improved training done.")
print("  Next: python 07_debiasing_reweighing.py (uses new models)")
print("=" * 65 + "\n")
