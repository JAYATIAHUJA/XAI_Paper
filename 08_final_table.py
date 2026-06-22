"""
=============================================================
STEP 8 -- Final Results Table (Paper-Ready)
=============================================================
Input  : All previous pkl files
Output : outputs/final_results_table.png  +  CSV
=============================================================
Run:
    python 08_final_table.py
=============================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)

os.makedirs("outputs", exist_ok=True)

print("\n" + "=" * 60)
print("  STEP 8 -- FINAL PAPER RESULTS TABLE")
print("=" * 60)

# ── Load all results ──────────────────────────────────────
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)

with open("data/models.pkl", "rb") as f:
    mdata = pickle.load(f)

with open("data/fairness_results.pkl", "rb") as f:
    fairness_results = pickle.load(f)

with open("data/cfvr_results.pkl", "rb") as f:
    cfvr_data = pickle.load(f)

with open("data/debiased_model.pkl", "rb") as f:
    debiased = pickle.load(f)

y_test      = data["y_test"]
gender_test = data["gender_test"]
X_test      = data["X_test"]
feature_names = data["feature_names"]

cfvr_results = cfvr_data["cfvr_results"]

# ── Compute gender_idx for CFVR ───────────────────────────
gender_idx = next(
    (i for i, n in enumerate(feature_names) if "gender" in n.lower()), None
)

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
    delta = np.abs(model.predict_proba(X_cf)[:, 1] - p_orig)
    return (delta > threshold).mean()


# ── Helper to gather all metrics for a model ─────────────
def get_all_metrics(y_test, y_pred, y_proba, gender_test, model=None, X_test=None, gender_idx=None):
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    spd  = demographic_parity_difference(y_test, y_pred, sensitive_features=gender_test)
    try:
        dir_ = demographic_parity_ratio(y_test, y_pred, sensitive_features=gender_test)
    except Exception:
        dir_ = float("nan")
    eod  = equalized_odds_difference(y_test, y_pred, sensitive_features=gender_test)
    cfvr = compute_cfvr(model, X_test, gender_idx) if (model is not None and gender_idx is not None) else float("nan")
    return {
        "Accuracy": round(acc, 4),
        "F1"      : round(f1, 4),
        "ROC-AUC" : round(auc, 4),
        "DIR"     : round(dir_, 4),
        "SPD"     : round(abs(spd), 4),    # report absolute value
        "EOD"     : round(eod, 4),
        "CFVR"    : round(cfvr, 4),
    }

# ── Build table rows ──────────────────────────────────────
rows = {}

for model_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
    preds  = mdata["predictions"][model_name]
    model  = mdata["trained_models"][model_name]
    rows[model_name] = get_all_metrics(
        y_test, preds["y_pred"], preds["y_proba"],
        gender_test, model, X_test, gender_idx
    )

# Debiased XGBoost (Reweighing)
rows["XGB + Reweighing"] = get_all_metrics(
    y_test, debiased["y_pred"], debiased["y_proba"],
    gender_test, debiased["model"], X_test, gender_idx
)

# Placeholder rows for paper completeness
rows["XGB + SHAP-Select"] = {k: "--" for k in ["Accuracy","F1","ROC-AUC","DIR","SPD","EOD","CFVR"]}
rows["CEFHF (Proposed)"]  = {k: "--" for k in ["Accuracy","F1","ROC-AUC","DIR","SPD","EOD","CFVR"]}

# ── Print table ───────────────────────────────────────────
print("\n" + "=" * 100)
print("  FINAL RESULTS TABLE (Paper-Ready)")
print("=" * 100)
cols = ["Accuracy", "F1", "ROC-AUC", "DIR", "SPD", "EOD", "CFVR"]
header = f"  {'Model':<25}" + "".join(f"{c:>12}" for c in cols)
print(header)
print("  " + "-" * 95)
for model_name, metrics in rows.items():
    row = f"  {model_name:<25}" + "".join(
        f"{metrics[c]:>12}" if isinstance(metrics[c], str) else f"{metrics[c]:>12.4f}"
        for c in cols
    )
    print(row)
print("=" * 100)
print("\n  NOTE: SPD reported as |SPD| (absolute value)")
print("  NOTE: DIR ideal = 1.0 | SPD/EOD ideal = 0.0 | CFVR ideal = 0.0")
print("  NOTE: XGB+SHAP-Select and CEFHF rows reserved for future work")

# ── Save CSV ──────────────────────────────────────────────
final_df = pd.DataFrame(rows).T
final_df.index.name = "Model"
final_df.to_csv("outputs/final_results_table.csv")
print("\n[SAVED] outputs/final_results_table.csv")

# ── Plot -- Paper-quality table image ─────────────────────
numeric_rows = {k: v for k, v in rows.items()
                if all(isinstance(x, float) for x in v.values())}
plot_df = pd.DataFrame(numeric_rows).T[cols]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Final Results: Accuracy vs Fairness Trade-off",
             fontsize=14, fontweight="bold")

# Performance subplot
perf_cols = ["Accuracy", "F1", "ROC-AUC"]
x = np.arange(len(perf_cols))
colors_list = ["#4C72B0", "#55A868", "#C44E52", "#9467BD"]
for i, (model_name, _) in enumerate(numeric_rows.items()):
    vals = [plot_df.loc[model_name, c] for c in perf_cols]
    axes[0].bar(x + i * 0.2, vals, 0.18,
                label=model_name, color=colors_list[i % len(colors_list)],
                alpha=0.85, edgecolor="white")
axes[0].set_xticks(x + 0.3); axes[0].set_xticklabels(perf_cols)
axes[0].set_ylim(0.5, 1.05); axes[0].set_title("Performance Metrics")
axes[0].set_ylabel("Score"); axes[0].legend(fontsize=8)

# Fairness subplot
fair_cols = ["SPD", "EOD", "CFVR"]
x2 = np.arange(len(fair_cols))
for i, (model_name, _) in enumerate(numeric_rows.items()):
    vals = [plot_df.loc[model_name, c] for c in fair_cols]
    axes[1].bar(x2 + i * 0.2, vals, 0.18,
                label=model_name, color=colors_list[i % len(colors_list)],
                alpha=0.85, edgecolor="white")
axes[1].set_xticks(x2 + 0.3); axes[1].set_xticklabels(fair_cols)
axes[1].axhline(0.1, color="red", linestyle="--", linewidth=1.5,
                label="Fairness threshold (0.1)")
axes[1].set_title("Fairness Metrics (Lower = Fairer)")
axes[1].set_ylabel("Score"); axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/final_results_table.png", dpi=150)
plt.close()
print("[SAVED] outputs/final_results_table.png")

print("\n[NEXT STEP] Run: python 09_adult_uci_validation.py")
print("=" * 60 + "\n")
