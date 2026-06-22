"""
=============================================================
STEP 6 -- CFVR: Counterfactual Fairness Violation Rate
=============================================================
Input  : data/processed.pkl  +  data/models.pkl
Output : outputs/cfvr_*.png  +  console CFVR score
=============================================================
Run:
    python 06_cfvr.py

CFVR -- YOUR NOVEL METRIC
─────────────────────────────────────────────────────────────
Algorithm:
  For each test sample i:
    1. p_orig = model.predict_proba(x_i)[1]      <- original
    2. x_cf   = flip Gender (Female↔Male)
    3. p_cf   = model.predict_proba(x_cf)[1]     <- counterfactual
    4. violation_i = |p_cf - p_orig| > threshold

  CFVR = sum(violations) / total_samples

Interpretation:
  CFVR = 0.25 -> 25% of candidates would get a materially
  different shortlisting probability just by changing gender.
  This is a direct measure of individual-level fairness.
─────────────────────────────────────────────────────────────
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

THRESHOLD = 0.10   # configurable -- |Δp| > threshold -> violation

print("\n" + "=" * 60)
print("  STEP 6 -- CFVR (Counterfactual Fairness Violation Rate)")
print("=" * 60)
print(f"\n[CONFIG] Violation threshold: |Δp| > {THRESHOLD}")

# ── Load ──────────────────────────────────────────────────
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)

with open("data/models.pkl", "rb") as f:
    mdata = pickle.load(f)

X_test        = data["X_test"]
y_test        = data["y_test"]
feature_names = data["feature_names"]
gender_test   = data["gender_test"]   # 0=Female, 1=Male

# ── Find Gender column index in X_test ────────────────────
gender_idx = None
for i, name in enumerate(feature_names):
    if "gender" in name.lower() or "Gender" in name:
        gender_idx = i
        break

if gender_idx is None:
    raise ValueError("[ERROR] Cannot find Gender feature index. Check feature_names.")

print(f"[INFO] Gender feature index: {gender_idx} -> '{feature_names[gender_idx]}'")
print(f"[INFO] Test samples: {len(X_test)}")

# ── CFVR Computation ──────────────────────────────────────
def compute_cfvr(model, X_test, gender_idx, threshold=0.10):
    """
    Returns:
        cfvr         : float -- violation rate
        violations   : np.array -- bool mask of violated samples
        delta_proba  : np.array -- |p_cf - p_orig| for each sample
        p_orig       : np.array -- original probabilities
        p_cf         : np.array -- counterfactual probabilities
    """
    # Original probabilities
    p_orig = model.predict_proba(X_test)[:, 1]

    # Flip Gender: 0->1 or 1->0  (gender column is first scaled via StandardScaler)
    # Since StandardScaler maps Male(1)->+z and Female(0)->-z,
    # we need to find what the encoded values are for Male/Female in scaled space.
    # Approach: just swap the Gender column value with the other group's value.
    X_cf = X_test.copy()
    # Get unique gender values that appear in the test set
    unique_vals = np.unique(X_cf[:, gender_idx])
    if len(unique_vals) == 2:
        val_a, val_b = unique_vals[0], unique_vals[1]
        mask_a = X_cf[:, gender_idx] == val_a
        mask_b = X_cf[:, gender_idx] == val_b
        X_cf[mask_a, gender_idx] = val_b
        X_cf[mask_b, gender_idx] = val_a
    else:
        # fallback: negate
        X_cf[:, gender_idx] = -X_cf[:, gender_idx]

    # Counterfactual probabilities
    p_cf = model.predict_proba(X_cf)[:, 1]

    # Violations
    delta_proba = np.abs(p_cf - p_orig)
    violations  = delta_proba > threshold
    cfvr        = violations.mean()

    return cfvr, violations, delta_proba, p_orig, p_cf

# ── Run on all models ─────────────────────────────────────
cfvr_results = {}
cfvr_details = {}

for name, model in mdata["trained_models"].items():
    cfvr, violations, delta, p_orig, p_cf = compute_cfvr(
        model, X_test, gender_idx, THRESHOLD
    )
    cfvr_results[name] = round(cfvr, 4)
    cfvr_details[name] = {
        "violations" : violations,
        "delta"      : delta,
        "p_orig"     : p_orig,
        "p_cf"       : p_cf,
        "cfvr"       : cfvr,
    }
    n_total    = len(X_test)
    n_viol     = violations.sum()
    mean_delta = delta.mean()

    print(f"\n[MODEL] {name}")
    print(f"  Total samples  : {n_total}")
    print(f"  Violations     : {n_viol}")
    print(f"  CFVR           : {cfvr:.4f}  ({cfvr*100:.1f}%)")
    print(f"  Mean |Δp|      : {mean_delta:.4f}")
    if cfvr > 0.15:
        print(f"  [WARN]  HIGH CFVR -- serious individual-level fairness violation")
    elif cfvr > 0.05:
        print(f"  [WARN]  MODERATE CFVR -- individual fairness concerns exist")
    else:
        print(f"  [OK]  LOW CFVR -- model is relatively individually fair")

# ── Detailed analysis -- XGBoost ───────────────────────────
print("\n" + "=" * 55)
print("  DETAILED CFVR ANALYSIS -- XGBoost")
print("=" * 55)
xgb_detail = cfvr_details["XGBoost"]
delta      = xgb_detail["delta"]
p_orig     = xgb_detail["p_orig"]
p_cf       = xgb_detail["p_cf"]
violations = xgb_detail["violations"]

# Who is being harmed?
female_mask = (gender_test == 0)
male_mask   = (gender_test == 1)

female_viol_rate = violations[female_mask].mean()
male_viol_rate   = violations[male_mask].mean()
print(f"\n  Violation rate -- Female candidates: {female_viol_rate:.3f}")
print(f"  Violation rate -- Male candidates  : {male_viol_rate:.3f}")
print(f"\n  Among violations ({violations.sum()} samples):")
print(f"    Female candidates harmed (p_cf > p_orig): "
      f"{((p_cf > p_orig) & violations & female_mask).sum()}")
print(f"    Male candidates harmed   (p_cf > p_orig): "
      f"{((p_cf > p_orig) & violations & male_mask).sum()}")

# ── Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("CFVR -- Counterfactual Fairness Violation Rate Analysis (XGBoost)",
             fontsize=13, fontweight="bold")

# 1. Delta distribution
axes[0, 0].hist(delta, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
axes[0, 0].axvline(THRESHOLD, color="red", linestyle="--",
                   linewidth=2, label=f"Threshold = {THRESHOLD}")
axes[0, 0].set_title("|Δp| Distribution (Original vs Counterfactual)")
axes[0, 0].set_xlabel("|p_cf - p_orig|")
axes[0, 0].set_ylabel("Count")
axes[0, 0].legend()

# 2. Scatter: p_orig vs p_cf
scatter_colors = np.where(violations, "#C44E52", "#4C72B0")
axes[0, 1].scatter(p_orig, p_cf, c=scatter_colors, alpha=0.4, s=15)
axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1.5, label="No change line")
axes[0, 1].set_title("P(Shortlisted) -- Original vs Counterfactual")
axes[0, 1].set_xlabel("P(Original)")
axes[0, 1].set_ylabel("P(Counterfactual = Gender Flipped)")
from matplotlib.patches import Patch
legend_els = [Patch(color="#C44E52", label="Violation"),
              Patch(color="#4C72B0", label="No Violation")]
axes[0, 1].legend(handles=legend_els)

# 3. CFVR comparison bar chart
names_list = list(cfvr_results.keys())
vals_list  = [cfvr_results[n] for n in names_list]
bars = axes[1, 0].bar(names_list, vals_list,
                      color=["#4C72B0", "#55A868", "#C44E52"],
                      alpha=0.85, edgecolor="white")
axes[1, 0].axhline(0.15, color="orange", linestyle="--", linewidth=1.5,
                   label="High CFVR threshold (0.15)")
axes[1, 0].set_title("CFVR by Model")
axes[1, 0].set_ylabel("CFVR")
axes[1, 0].set_ylim(0, min(max(vals_list) * 1.3, 1.0))
axes[1, 0].legend()
for bar, val in zip(bars, vals_list):
    axes[1, 0].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

# 4. Group-wise violation rates
group_labels = ["Female\n(Gender=0)", "Male\n(Gender=1)"]
group_rates  = [female_viol_rate, male_viol_rate]
bars2 = axes[1, 1].bar(group_labels, group_rates,
                        color=["#DD8452", "#4C72B0"], alpha=0.85, edgecolor="white")
axes[1, 1].set_title("CFVR by Gender Group (XGBoost)")
axes[1, 1].set_ylabel("Violation Rate")
axes[1, 1].set_ylim(0, 1.0)
for bar, val in zip(bars2, group_rates):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/cfvr_analysis.png", dpi=150)
plt.close()
print("\n[SAVED] outputs/cfvr_analysis.png")

# ── Summary table ─────────────────────────────────────────
print("\n" + "=" * 40)
print("  CFVR SUMMARY")
print("=" * 40)
for name, val in cfvr_results.items():
    print(f"  {name:<22} CFVR = {val:.4f}")

# ── Save ──────────────────────────────────────────────────
with open("data/cfvr_results.pkl", "wb") as f:
    pickle.dump({"cfvr_results": cfvr_results, "cfvr_details": cfvr_details}, f)
print("\n[SUCCESS] Saved data/cfvr_results.pkl")
print("\n[NEXT STEP] Run: python 07_debiasing_reweighing.py")
print("=" * 60 + "\n")
