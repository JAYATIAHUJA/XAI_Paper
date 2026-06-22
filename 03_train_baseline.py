"""
=============================================================
STEP 3 -- Train Baseline Models (LR, RF, XGB)
=============================================================
Input  : data/processed.pkl
Output : data/models.pkl  +  outputs/baseline_metrics.png
=============================================================
Run:
    python 03_train_baseline.py
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

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (accuracy_score, precision_score,
                                    recall_score, f1_score, roc_auc_score,
                                    classification_report, confusion_matrix,
                                    ConfusionMatrixDisplay)
from xgboost import XGBClassifier

os.makedirs("outputs", exist_ok=True)

print("\n" + "=" * 60)
print("  STEP 3 -- BASELINE MODEL TRAINING")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────
with open("data/processed.pkl", "rb") as f:
    data = pickle.load(f)

X_train = data["X_train"]
X_test  = data["X_test"]
y_train = data["y_train"]
y_test  = data["y_test"]

print(f"\n[INFO] Train: {X_train.shape} | Test: {X_test.shape}")
print(f"[INFO] Label balance (test): {np.bincount(y_test)}")

# ── Define models ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, eval_metric="logloss",
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        verbosity=0
    ),
}

# ── Evaluate ──────────────────────────────────────────────
def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "Accuracy" : round(accuracy_score(y_te, y_pred), 4),
        "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_te, y_pred, zero_division=0), 4),
        "F1"       : round(f1_score(y_te, y_pred, zero_division=0), 4),
        "ROC-AUC"  : round(roc_auc_score(y_te, y_proba), 4),
    }
    return model, y_pred, y_proba, metrics

results = {}
trained_models = {}
predictions    = {}

for name, model in models.items():
    print(f"\n[TRAINING] {name}...")
    trained_model, y_pred, y_proba, metrics = evaluate(
        name, model, X_train, y_train, X_test, y_test
    )
    results[name]         = metrics
    trained_models[name]  = trained_model
    predictions[name]     = {"y_pred": y_pred, "y_proba": y_proba}
    print(f"  Accuracy : {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall   : {metrics['Recall']:.4f}")
    print(f"  F1       : {metrics['F1']:.4f}")
    print(f"  ROC-AUC  : {metrics['ROC-AUC']:.4f}")

# ── Print results table ───────────────────────────────────
print("\n" + "=" * 65)
print("  BASELINE RESULTS TABLE")
print("=" * 65)
results_df = pd.DataFrame(results).T
results_df.index.name = "Model"
print(results_df.to_string())

# ── Save as CSV ───────────────────────────────────────────
results_df.to_csv("outputs/baseline_metrics.csv")
print("\n[SAVED] outputs/baseline_metrics.csv")

# ── Plot -- metrics comparison ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Baseline Model Performance Comparison", fontsize=14, fontweight="bold")

metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
x = np.arange(len(metric_cols))
width = 0.25
colors = ["#4C72B0", "#55A868", "#C44E52"]

for i, (name, metrics) in enumerate(results.items()):
    vals = [metrics[m] for m in metric_cols]
    axes[0].bar(x + i * width, vals, width, label=name,
                color=colors[i], alpha=0.85, edgecolor="white")

axes[0].set_xticks(x + width)
axes[0].set_xticklabels(metric_cols, rotation=15)
axes[0].set_ylim(0, 1.1)
axes[0].set_title("Metrics by Model")
axes[0].set_ylabel("Score")
axes[0].legend()
for ax in [axes[0]]:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)

# ROC-AUC comparison bar
auc_vals = [results[m]["ROC-AUC"] for m in results]
bars = axes[1].barh(list(results.keys()), auc_vals,
                     color=colors, alpha=0.85, edgecolor="white")
axes[1].set_xlim(0, 1.0)
axes[1].set_title("ROC-AUC Comparison")
axes[1].set_xlabel("ROC-AUC Score")
for bar, val in zip(bars, auc_vals):
    axes[1].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/baseline_metrics.png", dpi=150)
plt.close()
print("[SAVED] outputs/baseline_metrics.png")

# ── Confusion matrices ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
for ax, (name, preds) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, preds["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=150)
plt.close()
print("[SAVED] outputs/confusion_matrices.png")

# ── Save models & predictions ─────────────────────────────
models_bundle = {
    "trained_models": trained_models,
    "predictions"   : predictions,
    "results"       : results,
}
with open("data/models.pkl", "wb") as f:
    pickle.dump(models_bundle, f)
print("\n[SUCCESS] Saved data/models.pkl")
print("\n[NEXT STEP] Run: python 04_shap_analysis.py")
print("=" * 60 + "\n")
