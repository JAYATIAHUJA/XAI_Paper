"""
=============================================================
STEP 5 -- Fairness Metrics (SPD, DIR, EOD)
=============================================================
Input  : data/processed.pkl  +  data/models.pkl
Output : outputs/fairness_metrics.png  +  console table
=============================================================
Run:
    python 05_fairness_metrics.py

METRICS:
  SPD (Statistical Parity Difference) = P(ŷ=1|Female) - P(ŷ=1|Male)
       -> Ideal: 0.0  |  Unfair if |SPD| > 0.1
  DIR (Disparate Impact Ratio)        = P(ŷ=1|Female) / P(ŷ=1|Male)
       -> Ideal: 1.0  |  Unfair if DIR < 0.8 (80% rule)
  EOD (Equalized Odds Difference)     = max(|TPR_f-TPR_m|, |FPR_f-FPR_m|)
       -> Ideal: 0.0  |  Unfair if EOD > 0.1
=============================================================
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

from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)

os.makedirs("outputs", exist_ok=True)

print("\n" + "=" * 60)
print("  STEP 5 -- FAIRNESS METRICS (SPD / DIR / EOD)")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)

with open("data/models.pkl", "rb") as f:
    mdata = pickle.load(f)

y_test      = data["y_test"]
gender_test = data["gender_test"]   # 0=Female, 1=Male

# ── Compute metrics for all models ───────────────────────
def compute_fairness(y_true, y_pred, sensitive):
    """sensitive: array of 0/1 (0=Female=unprivileged, 1=Male=privileged)"""
    spd = demographic_parity_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    )
    try:
        dir_val = demographic_parity_ratio(
            y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
        )
    except Exception:
        dir_val = float("nan")
    eod = equalized_odds_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    )
    return {
        "SPD": round(spd, 4),
        "DIR": round(dir_val, 4),
        "EOD": round(eod, 4),
    }

fairness_results = {}

for model_name, preds in mdata["predictions"].items():
    y_pred = preds["y_pred"]
    fm     = compute_fairness(y_test, y_pred, gender_test)
    fairness_results[model_name] = fm

    print(f"\n[MODEL] {model_name}")
    print(f"  SPD (Statistical Parity Difference) : {fm['SPD']:+.4f}")
    print(f"  DIR (Disparate Impact Ratio)        :  {fm['DIR']:.4f}")
    print(f"  EOD (Equalized Odds Difference)     :  {fm['EOD']:.4f}")

    # Interpretation
    if abs(fm["SPD"]) > 0.1:
        print(f"  [WARN]  SPD > 0.1 -> Demographic parity violated")
    if fm["DIR"] < 0.8:
        print(f"  [WARN]  DIR < 0.8 -> 80% disparate impact rule violated")
    if fm["EOD"] > 0.1:
        print(f"  [WARN]  EOD > 0.1 -> Equalized odds violated")

# ── Print table ───────────────────────────────────────────
print("\n" + "=" * 60)
print("  FAIRNESS METRICS SUMMARY TABLE")
print("=" * 60)
fm_df = pd.DataFrame(fairness_results).T
fm_df.index.name = "Model"
print(fm_df.to_string())
fm_df.to_csv("outputs/fairness_metrics.csv")
print("\n[SAVED] outputs/fairness_metrics.csv")

# ── Detailed group-level analysis (XGBoost) ───────────────
print("\n[DETAIL] Group-level analysis -- XGBoost")
xgb_pred = mdata["predictions"]["XGBoost"]["y_pred"]

female_mask = (gender_test == 0)
male_mask   = (gender_test == 1)

female_sr = xgb_pred[female_mask].mean()
male_sr   = xgb_pred[male_mask].mean()

female_tpr = ((xgb_pred[female_mask] == 1) & (y_test[female_mask] == 1)).sum() / max((y_test[female_mask] == 1).sum(), 1)
male_tpr   = ((xgb_pred[male_mask]   == 1) & (y_test[male_mask]   == 1)).sum() / max((y_test[male_mask]   == 1).sum(), 1)

female_fpr = ((xgb_pred[female_mask] == 1) & (y_test[female_mask] == 0)).sum() / max((y_test[female_mask] == 0).sum(), 1)
male_fpr   = ((xgb_pred[male_mask]   == 1) & (y_test[male_mask]   == 0)).sum() / max((y_test[male_mask]   == 0).sum(), 1)

print(f"\n  Group          | Female   | Male")
print(f"  ─────────────────────────────────")
print(f"  Selection Rate | {female_sr:.3f}    | {male_sr:.3f}")
print(f"  TPR (Recall)   | {female_tpr:.3f}    | {male_tpr:.3f}")
print(f"  FPR            | {female_fpr:.3f}    | {male_fpr:.3f}")
print(f"\n  Interpretation:")
print(f"  Selection Rate gap = {male_sr - female_sr:+.3f}")
print(f"  TPR gap            = {male_tpr - female_tpr:+.3f}")
print(f"  FPR gap            = {male_fpr - female_fpr:+.3f}")

# ── Plot ──────────────────────────────────────────────────
metrics_list = ["SPD", "DIR", "EOD"]
model_names  = list(fairness_results.keys())
x = np.arange(len(metrics_list))
width = 0.25
colors = ["#4C72B0", "#55A868", "#C44E52"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Fairness Metrics Comparison Across Models", fontsize=14, fontweight="bold")

for i, name in enumerate(model_names):
    vals = [fairness_results[name][m] for m in metrics_list]
    axes[0].bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85, edgecolor="white")

axes[0].axhline(0.0,  color="green", linestyle="--", linewidth=1.5, label="Ideal SPD/EOD=0")
axes[0].axhline(0.1,  color="orange", linestyle=":",  linewidth=1.2, label="Fairness threshold ±0.1")
axes[0].axhline(-0.1, color="orange", linestyle=":",  linewidth=1.2)
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metrics_list)
axes[0].set_title("SPD, DIR, EOD by Model")
axes[0].set_ylabel("Metric Value")
axes[0].legend(fontsize=8)

# Group rates plot
groups   = ["Female", "Male"]
sr_vals  = [female_sr, male_sr]
tpr_vals = [female_tpr, male_tpr]
fpr_vals = [female_fpr, male_fpr]

x2 = np.arange(len(groups))
axes[1].bar(x2 - 0.25, sr_vals,  0.2, label="Selection Rate", color="#4C72B0", alpha=0.85)
axes[1].bar(x2,         tpr_vals, 0.2, label="TPR",            color="#55A868", alpha=0.85)
axes[1].bar(x2 + 0.25,  fpr_vals, 0.2, label="FPR",            color="#C44E52", alpha=0.85)
axes[1].set_xticks(x2)
axes[1].set_xticklabels(groups)
axes[1].set_title("XGBoost -- Group-Level Rates")
axes[1].set_ylabel("Rate")
axes[1].set_ylim(0, 1.1)
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/fairness_metrics.png", dpi=150)
plt.close()
print("[SAVED] outputs/fairness_metrics.png")

# ── Save fairness results ─────────────────────────────────
with open("data/fairness_results.pkl", "wb") as f:
    pickle.dump(fairness_results, f)
print("[SUCCESS] Saved data/fairness_results.pkl")
print("\n[NEXT STEP] Run: python 06_cfvr.py")
print("=" * 60 + "\n")
