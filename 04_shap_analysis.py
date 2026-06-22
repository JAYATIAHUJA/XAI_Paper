"""
=============================================================
STEP 4 -- SHAP Explainability
=============================================================
Input  : data/processed.pkl  +  data/models.pkl
Output : outputs/shap_*.png
=============================================================
Run:
    python 04_shap_analysis.py

SHAP = SHapley Additive exPlanations
Each feature gets a SHAP value for each prediction.
Positive SHAP -> pushes prediction toward Shortlisted=1
Negative SHAP -> pushes prediction toward Shortlisted=0
=============================================================
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)

print("\n" + "=" * 60)
print("  STEP 4 -- SHAP ANALYSIS")
print("=" * 60)

# ── Load data and models ──────────────────────────────────
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)

with open("data/models.pkl", "rb") as f:
    mdata = pickle.load(f)

X_train       = data["X_train"]
X_test        = data["X_test"]
feature_names = data["feature_names"]
xgb_model     = mdata["trained_models"]["XGBoost"]

print(f"\n[INFO] X_test shape    : {X_test.shape}")
print(f"[INFO] Feature count   : {len(feature_names)}")
print(f"[INFO] Computing SHAP values for XGBoost...")

# ── SHAP TreeExplainer (fast for tree-based models) ───────
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

print(f"[INFO] SHAP values shape: {shap_values.shape}")

# ── 1. SHAP Summary Plot (beeswarm) ───────────────────────
print("\n[PLOT] Generating SHAP summary beeswarm plot...")
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    show=False,
    max_display=15,
    plot_type="dot"
)
plt.title("SHAP Summary Plot -- XGBoost\n(Red = High value, Blue = Low value)",
          fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("outputs/shap_01_summary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] outputs/shap_01_summary_beeswarm.png")

# ── 2. SHAP Bar Plot (mean |SHAP|) ────────────────────────
print("[PLOT] Generating SHAP mean absolute bar plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    show=False,
    max_display=15,
    plot_type="bar"
)
plt.title("SHAP Feature Importance (Mean |SHAP Value|)",
          fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("outputs/shap_02_bar_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] outputs/shap_02_bar_importance.png")

# ── 3. Gender-specific SHAP analysis ─────────────────────
print("\n[ANALYSIS] Gender feature SHAP impact...")
gender_idx = None
for i, name in enumerate(feature_names):
    if "gender" in name.lower() or "Gender" in name:
        gender_idx = i
        break

if gender_idx is not None:
    gender_shap = shap_values[:, gender_idx]
    print(f"  Feature name         : '{feature_names[gender_idx]}'")
    print(f"  Mean SHAP (Gender)   : {gender_shap.mean():.4f}")
    print(f"  Std  SHAP (Gender)   : {gender_shap.std():.4f}")
    print(f"  Max  SHAP (Gender)   : {gender_shap.max():.4f}")
    print(f"  Min  SHAP (Gender)   : {gender_shap.min():.4f}")
    print(f"\n  Interpretation: A positive mean SHAP for Gender means")
    print(f"  Male (encoded=1) is pushing predictions toward Shortlisted=1")
    print(f"  i.e., Gender is CONTRIBUTING TO BIAS in the model.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gender_shap[gender_shap > 0], bins=30, alpha=0.7,
            color="#4C72B0", label="Positive SHAP (-> Shortlisted)")
    ax.hist(gender_shap[gender_shap < 0], bins=30, alpha=0.7,
            color="#DD8452", label="Negative SHAP (-> Not Shortlisted)")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(f"SHAP Values for '{feature_names[gender_idx]}' Feature",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("SHAP Value")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/shap_03_gender_impact.png", dpi=150)
    plt.close()
    print("[SAVED] outputs/shap_03_gender_impact.png")
else:
    print("  [WARN] Gender feature index not found.")

# ── 4. Top features ranked by mean |SHAP| ────────────────
mean_shap = np.abs(shap_values).mean(axis=0)
ranked    = sorted(zip(feature_names, mean_shap), key=lambda x: x[1], reverse=True)

print("\n" + "=" * 55)
print("  TOP FEATURES BY MEAN |SHAP VALUE|")
print("=" * 55)
print(f"  {'Rank':<5} {'Feature':<30} {'Mean |SHAP|'}")
print("  " + "-" * 50)
for rank, (feat, val) in enumerate(ranked[:15], 1):
    marker = " <- GENDER" if ("gender" in feat.lower() or "Gender" in feat) else ""
    print(f"  {rank:<5} {feat:<30} {val:.4f}{marker}")

# ── Save SHAP values ──────────────────────────────────────
shap_bundle = {
    "shap_values"  : shap_values,
    "feature_names": feature_names,
    "explainer"    : explainer,
    "mean_shap"    : mean_shap,
    "ranked"       : ranked,
}
with open("data/shap_values.pkl", "wb") as f:
    pickle.dump(shap_bundle, f)
print("\n[SUCCESS] Saved data/shap_values.pkl")
print("\n[NEXT STEP] Run: python 05_fairness_metrics.py")
print("=" * 60 + "\n")
