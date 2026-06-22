"""
=============================================================
EXPERIMENT: Bias vs Instability Check
=============================================================
Key question your supervisor asked:

  Is CFVR = 0.385 because the model is BIASED against gender?
  Or is it because the model is UNSTABLE (everything causes big shifts)?

Method:
  Flip each feature individually across all test samples.
  Measure the mean |Δp| and violation rate for each flip.

If Gender >> others  → strong evidence of bias
If all features ≈    → model instability, not just gender bias

Also includes:
  - Confusion matrices (supervisor request)
  - Class distribution (supervisor request)
  - Gender-wise actual vs predicted selection rates
  - Statistical significance test (McNemar)
=============================================================
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

THRESHOLD = 0.10

print("\n" + "=" * 65)
print("  EXPERIMENT: BIAS VS MODEL INSTABILITY")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)
with open("data/models.pkl", "rb") as f:
    mdata = pickle.load(f)

X_test        = data["X_test"]
y_test        = data["y_test"]
feature_names = data["feature_names"]
gender_test   = data["gender_test"]

xgb_model = mdata["trained_models"]["XGBoost"]
rf_model  = mdata["trained_models"]["Random Forest"]

# ─────────────────────────────────────────────────────────
# PART 1: Class distribution & selection rates
# ─────────────────────────────────────────────────────────
print("\n[PART 1] DATASET SANITY CHECK")
print("-" * 45)
f_mask = gender_test == 0
m_mask = gender_test == 1

print(f"  Test set total    : {len(y_test)}")
print(f"  Shortlisted (=1)  : {y_test.sum()} ({y_test.mean()*100:.1f}%)")
print(f"  Not shortlisted   : {(1-y_test).sum()} ({(1-y_test.mean())*100:.1f}%)")
print()
print(f"  Female (n={f_mask.sum()}): Actual shortlisted rate = {y_test[f_mask].mean()*100:.1f}%")
print(f"  Male   (n={m_mask.sum()}): Actual shortlisted rate = {y_test[m_mask].mean()*100:.1f}%")
print(f"  True SPD (ground truth) = {abs(y_test[f_mask].mean() - y_test[m_mask].mean())*100:.1f}pp")
print()

# Predicted rates
for model_name, preds in mdata["predictions"].items():
    y_pred = preds["y_pred"]
    f_rate = y_pred[f_mask].mean() * 100
    m_rate = y_pred[m_mask].mean() * 100
    print(f"  {model_name:22}: Female pred={f_rate:.1f}%  Male pred={m_rate:.1f}%  |Pred SPD|={abs(f_rate-m_rate):.1f}pp")

# ─────────────────────────────────────────────────────────
# PART 2: Feature Flip CFVR (Bias vs Instability)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PART 2: FEATURE FLIP COMPARISON")
print("  (Gender vs Age vs Experience vs Education)")
print("=" * 65)
print(f"\n  Threshold: |Δp| > {THRESHOLD}")
print(f"  Model: XGBoost")
print()

def compute_flip_cfvr(model, X_test, feature_idx, threshold=0.10, flip_mode="swap"):
    """
    Compute CFVR-like metric for any feature.
    flip_mode:
      'swap'   → swap unique values (for binary/categorical)
      'negate' → negate the feature value
      'shift_1sd' → shift by +1 standard deviation
      'shift_-1sd' → shift by -1 standard deviation
    """
    p_orig = model.predict_proba(X_test)[:, 1]
    X_cf = X_test.copy()

    if flip_mode == "swap":
        uniq = np.unique(X_cf[:, feature_idx])
        if len(uniq) == 2:
            val_a, val_b = uniq[0], uniq[1]
            mask_a = X_cf[:, feature_idx] == val_a
            mask_b = X_cf[:, feature_idx] == val_b
            X_cf[mask_a, feature_idx] = val_b
            X_cf[mask_b, feature_idx] = val_a
        else:
            # many values → flip by negating (mirror around mean=0 since scaled)
            X_cf[:, feature_idx] = -X_cf[:, feature_idx]
    elif flip_mode == "shift_1sd":
        X_cf[:, feature_idx] += 1.0   # +1 std in scaled space
    elif flip_mode == "shift_-1sd":
        X_cf[:, feature_idx] -= 1.0

    p_cf = model.predict_proba(X_cf)[:, 1]
    delta = np.abs(p_cf - p_orig)
    violations = delta > threshold
    return {
        "cfvr"      : violations.mean(),
        "mean_delta": delta.mean(),
        "max_delta" : delta.max(),
        "n_violated": violations.sum(),
    }

# Features to test
# Gender_enc=0, age=1, experience_years=2, screening_score=3, education_level_*=4-7
test_features = {
    "Gender (Gender_enc)"       : (0, "swap"),
    "Age (+1 SD)"               : (1, "shift_1sd"),
    "Age (−1 SD)"               : (1, "shift_-1sd"),
    "Experience (+1 SD)"        : (2, "shift_1sd"),
    "Experience (−1 SD)"        : (2, "shift_-1sd"),
    "Screening Score (+1 SD)"   : (3, "shift_1sd"),
    "Screening Score (−1 SD)"   : (3, "shift_-1sd"),
}

flip_results = {}
for label, (idx, mode) in test_features.items():
    r = compute_flip_cfvr(xgb_model, X_test, idx, THRESHOLD, mode)
    flip_results[label] = r
    flag = "  <-- GENDER" if "Gender" in label else ""
    print(f"  {label:35}: CFVR={r['cfvr']:.4f}  mean|Δp|={r['mean_delta']:.4f}  max|Δp|={r['max_delta']:.4f}{flag}")

# Interpretation
gender_cfvr = flip_results["Gender (Gender_enc)"]["cfvr"]
age_cfvr_hi = flip_results["Age (+1 SD)"]["cfvr"]
age_cfvr_lo = flip_results["Age (−1 SD)"]["cfvr"]
exp_cfvr_hi = flip_results["Experience (+1 SD)"]["cfvr"]
scr_cfvr_hi = flip_results["Screening Score (+1 SD)"]["cfvr"]

print()
print("  INTERPRETATION:")
if gender_cfvr > max(age_cfvr_hi, age_cfvr_lo, exp_cfvr_hi, scr_cfvr_hi):
    print("  ✅ Gender CFVR > all other features → Evidence of BIAS (not just instability)")
elif gender_cfvr > scr_cfvr_hi * 0.5:
    print("  ⚠️  Gender CFVR is elevated but Screening Score also high → Mixed signal")
else:
    print("  ❌ All features show similar CFVR → Evidence of MODEL INSTABILITY")

# ─────────────────────────────────────────────────────────
# PART 3: Controlled experiment – same magnitude flip
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PART 3: SAME-MAGNITUDE FLIP (±1 SD on all continuous features)")
print("  This ensures a fair comparison across features")
print("=" * 65)

continuous_features = {
    "Gender_enc"      : 0,  # swap = effectively ±1 SD in scaled space
    "Age"             : 1,
    "Experience"      : 2,
    "Screening Score" : 3,
}

results_pos = {}
results_neg = {}
for label, idx in continuous_features.items():
    if label == "Gender_enc":
        r = compute_flip_cfvr(xgb_model, X_test, idx, THRESHOLD, "swap")
        results_pos[label] = r
        results_neg[label] = r  # same for binary
    else:
        results_pos[label] = compute_flip_cfvr(xgb_model, X_test, idx, THRESHOLD, "shift_1sd")
        results_neg[label] = compute_flip_cfvr(xgb_model, X_test, idx, THRESHOLD, "shift_-1sd")

print(f"\n  {'Feature':20} | {'CFVR (+1SD)':12} | {'CFVR (−1SD)':12} | {'mean|Δp| +':12} | {'mean|Δp| −':12}")
print("  " + "-" * 75)
for label in continuous_features:
    rp = results_pos[label]
    rn = results_neg[label]
    flag = " ← GENDER" if label == "Gender_enc" else ""
    print(f"  {label:20} | {rp['cfvr']:>12.4f} | {rn['cfvr']:>12.4f} | {rp['mean_delta']:>12.4f} | {rn['mean_delta']:>12.4f}{flag}")

# ─────────────────────────────────────────────────────────
# PART 4: Statistical significance (McNemar test)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PART 4: MCNEMAR TEST (Statistical Significance)")
print("  H0: Predictions are identical for Female vs Male")
print("=" * 65)

p_orig = xgb_model.predict_proba(X_test)[:, 1]
# Flip gender
X_cf = X_test.copy()
uniq = np.unique(X_cf[:, 0])
val_a, val_b = uniq[0], uniq[1]
mask_a = X_cf[:, 0] == val_a
mask_b = X_cf[:, 0] == val_b
X_cf[mask_a, 0] = val_b
X_cf[mask_b, 0] = val_a
p_cf = xgb_model.predict_proba(X_cf)[:, 1]

pred_orig = (p_orig >= 0.5).astype(int)
pred_cf   = (p_cf >= 0.5).astype(int)

# McNemar contingency table
b = ((pred_orig == 1) & (pred_cf == 0)).sum()  # orig=1, cf=0
c = ((pred_orig == 0) & (pred_cf == 1)).sum()  # orig=0, cf=1
n_discordant = b + c

print(f"\n  Discordant pairs where prediction flipped:")
print(f"    orig=1, cf=0 (downgraded by gender flip): {b}")
print(f"    orig=0, cf=1 (upgraded by gender flip)  : {c}")
print(f"    Total discordant: {n_discordant} / {len(y_test)} ({n_discordant/len(y_test)*100:.1f}%)")

if n_discordant >= 5:
    # McNemar statistic (with continuity correction)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = stats.chi2.sf(chi2, df=1)
    print(f"\n  McNemar chi²  = {chi2:.4f}")
    print(f"  p-value       = {p_val:.6f}")
    if p_val < 0.001:
        print("  ✅ Statistically significant (p < 0.001)")
        print("  → Gender flip causes significantly different predictions.")
        print("  → CFVR is detecting real signal, not random noise.")
    elif p_val < 0.05:
        print("  ✅ Statistically significant (p < 0.05)")
    else:
        print("  ❌ NOT statistically significant (p > 0.05)")
        print("  → Cannot rule out that CFVR is due to random noise.")
else:
    print("  Too few discordant pairs for reliable McNemar test.")

# ─────────────────────────────────────────────────────────
# PART 5: Visualizations
# ─────────────────────────────────────────────────────────
print("\n[PLOTS] Generating...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Bias vs Model Instability — Feature Flip Analysis (XGBoost)",
             fontsize=14, fontweight="bold")

# 1. Feature flip CFVR comparison bar
labels = list(test_features.keys())
cfvr_vals = [flip_results[l]["cfvr"] for l in labels]
colors = ["#C44E52" if "Gender" in l else "#4C72B0" for l in labels]
bars = axes[0, 0].barh(labels, cfvr_vals, color=colors, alpha=0.85, edgecolor="white")
axes[0, 0].axvline(THRESHOLD, color="orange", linestyle="--", linewidth=1.5, label="Threshold")
axes[0, 0].set_xlabel("CFVR (Counterfactual Violation Rate)")
axes[0, 0].set_title("Feature Flip CFVR — Bias vs Instability")
axes[0, 0].legend(fontsize=9)
for bar, val in zip(bars, cfvr_vals):
    axes[0, 0].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9, fontweight="bold")
from matplotlib.patches import Patch
legend_els = [Patch(color="#C44E52", label="Gender"), Patch(color="#4C72B0", label="Other features")]
axes[0, 0].legend(handles=legend_els, fontsize=9)

# 2. Mean |Δp| comparison
mean_deltas = [flip_results[l]["mean_delta"] for l in labels]
bars2 = axes[0, 1].barh(labels, mean_deltas, color=colors, alpha=0.85, edgecolor="white")
axes[0, 1].set_xlabel("Mean |Δp| (avg probability shift)")
axes[0, 1].set_title("Mean Probability Shift per Feature Flip")
axes[0, 1].legend(handles=legend_els, fontsize=9)
for bar, val in zip(bars2, mean_deltas):
    axes[0, 1].text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

# 3. Prediction flip heatmap (orig vs cf)
flip_table = np.array([[((pred_orig == 0) & (pred_cf == 0)).sum(),
                         ((pred_orig == 0) & (pred_cf == 1)).sum()],
                        [((pred_orig == 1) & (pred_cf == 0)).sum(),
                         ((pred_orig == 1) & (pred_cf == 1)).sum()]])
im = axes[1, 0].imshow(flip_table, cmap="Blues")
axes[1, 0].set_xticks([0, 1]); axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(["CF pred=0", "CF pred=1"])
axes[1, 0].set_yticklabels(["Orig pred=0", "Orig pred=1"])
axes[1, 0].set_title("Prediction Agreement: Original vs Gender-Flipped")
for i in range(2):
    for j in range(2):
        axes[1, 0].text(j, i, str(flip_table[i, j]),
                        ha="center", va="center", fontsize=14, fontweight="bold",
                        color="white" if flip_table[i, j] > flip_table.max() / 2 else "black")
plt.colorbar(im, ax=axes[1, 0])

# 4. Same-magnitude comparison (±1 SD)
feat_labels = list(continuous_features.keys())
x = np.arange(len(feat_labels))
w = 0.35
cfvr_pos = [results_pos[l]["cfvr"] for l in feat_labels]
cfvr_neg = [results_neg[l]["cfvr"] for l in feat_labels]
bars_pos = axes[1, 1].bar(x - w/2, cfvr_pos, w, label="+1 SD / Swap",
                           color="#4C72B0", alpha=0.85, edgecolor="white")
bars_neg = axes[1, 1].bar(x + w/2, cfvr_neg, w, label="−1 SD / Swap",
                           color="#55A868", alpha=0.85, edgecolor="white")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(feat_labels, rotation=15, ha="right")
axes[1, 1].set_ylabel("CFVR")
axes[1, 1].set_title("Same-Magnitude Flip: Gender vs Other Features")
axes[1, 1].axhline(THRESHOLD, color="red", linestyle="--", linewidth=1.5,
                   label="Threshold (0.10)")
axes[1, 1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("outputs/bias_vs_instability.png", dpi=150)
plt.close()
print("[SAVED] outputs/bias_vs_instability.png")

print("\n" + "=" * 65)
print("  SUMMARY FOR PAPER")
print("=" * 65)
print()
print("  This experiment answers: Is CFVR detecting BIAS or INSTABILITY?")
print()
print("  Evidence for BIAS  → Gender flip CFVR >> other feature flips")
print("  Evidence for NOISE → All features show similar CFVR")
print()
print("  Significance test confirms whether CFVR is statistically real.")
print()
print("[COMPLETE]")
print("=" * 65 + "\n")
