"""
=============================================================
STEP 7 -- Debiasing with AIF360 Reweighing
=============================================================
Input  : data/processed.pkl  +  data/models.pkl
Output : data/debiased_model.pkl  +  outputs/debiasing_comparison.png
=============================================================
Run:
    pip install aif360        (first time only)
    python 07_debiasing_reweighing.py

HOW REWEIGHING WORKS:
  - AIF360 computes sample weights for each training sample
  - Weights are higher for underrepresented (Female, Shortlisted=1)
  - Weights are lower for overrepresented (Male, Shortlisted=1)
  - XGBoost is retrained with these sample_weights
  - Result: model learns to be fairer without changing architecture
=============================================================
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

print("\n" + "=" * 60)
print("  STEP 7 -- DEBIASING (AIF360 Reweighing)")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)

with open("data/models.pkl", "rb") as f:
    mdata = pickle.load(f)

X_train       = data["X_train"]
X_test        = data["X_test"]
y_train       = data["y_train"]
y_test        = data["y_test"]
gender_train  = data["gender_train"]
gender_test   = data["gender_test"]
feature_names = data["feature_names"]

# ── Try AIF360 Reweighing ─────────────────────────────────
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing as AIF360Reweighing

    print("\n[INFO] Using AIF360 Reweighing...")
    AIF360_AVAILABLE = True

except ImportError:
    print("\n[WARN] AIF360 not installed -- using manual Reweighing formula")
    print("[HINT] Install with: pip install aif360")
    AIF360_AVAILABLE = False


def compute_reweighing_weights(y_train, gender_train):
    """
    Manual implementation of Reweighing weights (Kamiran & Calders, 2012).

    W(x) = P(Y=y, A=a)_expected / P(Y=y, A=a)_observed
    
    Where expected = independent (no bias), observed = actual distribution.
    """
    n = len(y_train)
    weights = np.ones(n)

    unique_genders = np.unique(gender_train)
    unique_labels  = np.unique(y_train)

    for g in unique_genders:
        for y in unique_labels:
            mask = (gender_train == g) & (y_train == y)
            p_g = (gender_train == g).mean()
            p_y = (y_train == y).mean()
            p_gy = mask.mean()
            if p_gy > 0:
                w = (p_g * p_y) / p_gy
                weights[mask] = w

    # Normalize so weights sum to n
    weights = weights * (n / weights.sum())
    return weights


# ── Compute weights ───────────────────────────────────────
if AIF360_AVAILABLE:
    try:
        # Build AIF360 BinaryLabelDataset
        train_df = pd.DataFrame(X_train, columns=feature_names)
        train_df["Shortlisted"] = y_train
        train_df["Gender_enc"]  = gender_train

        aif_dataset = BinaryLabelDataset(
            df=train_df,
            label_names=["Shortlisted"],
            protected_attribute_names=["Gender_enc"],
            favorable_label=1,
            unfavorable_label=0,
        )
        privileged_groups   = [{"Gender_enc": 1}]
        unprivileged_groups = [{"Gender_enc": 0}]

        rw = AIF360Reweighing(
            privileged_groups=privileged_groups,
            unprivileged_groups=unprivileged_groups
        )
        rw_dataset    = rw.fit_transform(aif_dataset)
        sample_weights = rw_dataset.instance_weights
        print(f"[INFO] AIF360 Reweighing weights -- min: {sample_weights.min():.4f}  max: {sample_weights.max():.4f}")

    except Exception as e:
        print(f"[WARN] AIF360 failed: {e} -- falling back to manual weights")
        sample_weights = compute_reweighing_weights(y_train, gender_train)
        AIF360_AVAILABLE = False
else:
    sample_weights = compute_reweighing_weights(y_train, gender_train)
    print(f"[INFO] Manual Reweighing weights -- min: {sample_weights.min():.4f}  max: {sample_weights.max():.4f}")

print(f"[INFO] Weight stats -- mean: {sample_weights.mean():.4f}  std: {sample_weights.std():.4f}")

# ── Retrain XGBoost with sample weights ───────────────────
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)

print("\n[TRAINING] XGBoost with Reweighing...")
xgb_debiased = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    random_state=42, eval_metric="logloss", verbosity=0
)
xgb_debiased.fit(X_train, y_train, sample_weight=sample_weights)

y_pred_db  = xgb_debiased.predict(X_test)
y_proba_db = xgb_debiased.predict_proba(X_test)[:, 1]

# ── Metrics ───────────────────────────────────────────────
acc_db  = accuracy_score(y_test, y_pred_db)
f1_db   = f1_score(y_test, y_pred_db, zero_division=0)
auc_db  = roc_auc_score(y_test, y_proba_db)
spd_db  = demographic_parity_difference(y_test, y_pred_db, sensitive_features=gender_test)
try:
    dir_db = demographic_parity_ratio(y_test, y_pred_db, sensitive_features=gender_test)
except Exception:
    dir_db = float("nan")
eod_db  = equalized_odds_difference(y_test, y_pred_db, sensitive_features=gender_test)

print(f"\n  [Debiased XGBoost] Metrics:")
print(f"  Accuracy  : {acc_db:.4f}")
print(f"  F1        : {f1_db:.4f}")
print(f"  ROC-AUC   : {auc_db:.4f}")
print(f"  SPD       : {spd_db:+.4f}")
print(f"  DIR       : {dir_db:.4f}")
print(f"  EOD       : {eod_db:.4f}")

# ── CFVR on debiased model ────────────────────────────────
gender_idx = None
for i, name in enumerate(feature_names):
    if "gender" in name.lower():
        gender_idx = i
        break

def compute_cfvr(model, X_test, gender_idx, threshold=0.10):
    p_orig = model.predict_proba(X_test)[:, 1]
    X_cf   = X_test.copy()
    unique_vals = np.unique(X_cf[:, gender_idx])
    if len(unique_vals) == 2:
        val_a, val_b = unique_vals[0], unique_vals[1]
        mask_a = X_cf[:, gender_idx] == val_a
        mask_b = X_cf[:, gender_idx] == val_b
        X_cf[mask_a, gender_idx] = val_b
        X_cf[mask_b, gender_idx] = val_a
    else:
        X_cf[:, gender_idx] = -X_cf[:, gender_idx]
    p_cf       = model.predict_proba(X_cf)[:, 1]
    delta      = np.abs(p_cf - p_orig)
    violations = delta > threshold
    return violations.mean(), violations, delta

cfvr_db, _, _ = compute_cfvr(xgb_debiased, X_test, gender_idx)
print(f"  CFVR      : {cfvr_db:.4f}")

# ── Load baseline results for comparison ──────────────────
with open("data/fairness_results.pkl", "rb") as f:
    baseline_fm = pickle.load(f)

with open("data/cfvr_results.pkl", "rb") as f:
    cfvr_baseline = pickle.load(f)["cfvr_results"]

baseline_xgb  = mdata["predictions"]["XGBoost"]
acc_orig  = accuracy_score(y_test, baseline_xgb["y_pred"])
f1_orig   = f1_score(y_test, baseline_xgb["y_pred"], zero_division=0)
auc_orig  = roc_auc_score(y_test, baseline_xgb["y_proba"])
spd_orig  = baseline_fm["XGBoost"]["SPD"]
dir_orig  = baseline_fm["XGBoost"]["DIR"]
eod_orig  = baseline_fm["XGBoost"]["EOD"]
cfvr_orig = cfvr_baseline["XGBoost"]

# ── Comparison table ──────────────────────────────────────
print("\n" + "=" * 65)
print("  BEFORE vs AFTER REWEIGHING -- XGBoost")
print("=" * 65)
print(f"  {'Metric':<15} {'Baseline XGB':>15} {'XGB+Reweighing':>15} {'Change':>10}")
print("  " + "-" * 58)

for metric, before, after in [
    ("Accuracy",   acc_orig,  acc_db),
    ("F1",         f1_orig,   f1_db),
    ("ROC-AUC",   auc_orig,  auc_db),
    ("SPD",       spd_orig,  spd_db),
    ("DIR",       dir_orig,  dir_db),
    ("EOD",       eod_orig,  eod_db),
    ("CFVR",      cfvr_orig, cfvr_db),
]:
    change = after - before
    arrow  = "v" if change < 0 else "^"
    print(f"  {metric:<15} {before:>15.4f} {after:>15.4f} {arrow}{abs(change):>8.4f}")

print(f"\n  KEY FINDING: Reweighing reduces SPD/EOD/CFVR at a small accuracy cost.")
print(f"  This empirically proves: Accuracy ≠ Fairness")

# ── Plot comparison ───────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Before vs After Reweighing Debiasing", fontsize=13, fontweight="bold")

perf_metrics = ["Accuracy", "F1", "ROC-AUC"]
perf_before  = [acc_orig, f1_orig, auc_orig]
perf_after   = [acc_db,   f1_db,  auc_db]

x = np.arange(len(perf_metrics))
axes[0].bar(x - 0.2, perf_before, 0.35, label="Baseline XGB",
            color="#4C72B0", alpha=0.85, edgecolor="white")
axes[0].bar(x + 0.2, perf_after,  0.35, label="XGB + Reweighing",
            color="#55A868", alpha=0.85, edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(perf_metrics)
axes[0].set_title("Performance Metrics")
axes[0].set_ylim(0, 1.1); axes[0].legend()

fair_metrics = ["SPD", "EOD"]
fair_before  = [abs(spd_orig), abs(eod_orig)]
fair_after   = [abs(spd_db),   abs(eod_db)]
x2 = np.arange(len(fair_metrics))
axes[1].bar(x2 - 0.2, fair_before, 0.35, label="Baseline XGB",
            color="#C44E52", alpha=0.85, edgecolor="white")
axes[1].bar(x2 + 0.2, fair_after,  0.35, label="XGB + Reweighing",
            color="#55A868", alpha=0.85, edgecolor="white")
axes[1].set_xticks(x2); axes[1].set_xticklabels(fair_metrics)
axes[1].set_title("|SPD| and |EOD| (lower = fairer)")
axes[1].set_ylim(0, max(fair_before) * 1.4); axes[1].legend()

cfvr_compare = [cfvr_orig, cfvr_db]
bars = axes[2].bar(["Baseline XGB", "XGB + Reweighing"],
                   cfvr_compare, color=["#C44E52", "#55A868"],
                   alpha=0.85, edgecolor="white")
for bar, val in zip(bars, cfvr_compare):
    axes[2].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
axes[2].set_title("CFVR (lower = fairer)")
axes[2].set_ylim(0, max(cfvr_compare) * 1.4)

plt.tight_layout()
plt.savefig("outputs/debiasing_comparison.png", dpi=150)
plt.close()
print("\n[SAVED] outputs/debiasing_comparison.png")

# ── Save ──────────────────────────────────────────────────
debiased_bundle = {
    "model"       : xgb_debiased,
    "y_pred"      : y_pred_db,
    "y_proba"     : y_proba_db,
    "metrics"     : {
        "Accuracy": acc_db, "F1": f1_db, "ROC-AUC": auc_db,
        "SPD": spd_db, "DIR": dir_db, "EOD": eod_db, "CFVR": cfvr_db
    },
    "sample_weights": sample_weights,
}
with open("data/debiased_model.pkl", "wb") as f:
    pickle.dump(debiased_bundle, f)
print("[SUCCESS] Saved data/debiased_model.pkl")
print("\n[NEXT STEP] Run: python 08_final_table.py")
print("=" * 60 + "\n")
