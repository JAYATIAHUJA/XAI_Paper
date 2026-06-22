"""
=============================================================
13_STATISTICAL_SIGNIFICANCE.py -- Significance Testing
=============================================================
Input  : data/cv_pipeline_results.pkl
Output : outputs/significance_testing.csv
Runs paired t-tests comparing:
  - XGBoost (Tuned) vs XGBoost (Tuned + RW)
=============================================================
"""

import pickle
import numpy as np
import pandas as pd
from scipy import stats

# ── Load ──────────────────────────────────────────────────
with open("data/cv_pipeline_results.pkl", "rb") as f:
    bundle = pickle.load(f)

rec_metrics   = bundle["rec_cv_metrics"]
adult_metrics = bundle["adult_cv_metrics"]

models = ["XGBoost (Tuned)", "XGBoost (Tuned + RW)"]
metrics_list = ["Accuracy", "F1", "ROC-AUC", "DIR", "SPD", "EOD", "CFVR-Prob", "CFVR-Class"]

def perform_paired_ttest(cv_metrics, model_a, model_b, metrics):
    rows = []
    for m in metrics:
        vals_a = [fold_m[m] for fold_m in cv_metrics[model_a] if not np.isnan(fold_m[m])]
        vals_b = [fold_m[m] for fold_m in cv_metrics[model_b] if not np.isnan(fold_m[m])]
        
        # We need both to have exactly 5 elements for a paired t-test
        if len(vals_a) == 5 and len(vals_b) == 5:
            t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
            mean_a = np.mean(vals_a)
            mean_b = np.mean(vals_b)
            diff = mean_b - mean_a
            significant = "Yes *" if p_val < 0.05 else "No"
            rows.append({
                "Metric"      : m,
                "Mean (Tuned)": f"{mean_a:.4f}",
                "Mean (RW)"   : f"{mean_b:.4f}",
                "Difference"  : f"{diff:+.4f}",
                "t-statistic" : f"{t_stat:.4f}",
                "p-value"     : f"{p_val:.4e}",
                "Sig (p<0.05)": significant
            })
    return pd.DataFrame(rows)

print("\n" + "=" * 80)
# Recruitment Dataset significance
print("  SIGNIFICANCE TESTING (Paired t-test): XGBoost (Tuned) vs XGBoost (Tuned + RW)")
print("  Dataset: Recruitment Bias")
print("=" * 80)
rec_sig_df = perform_paired_ttest(rec_metrics, models[0], models[1], metrics_list)
print(rec_sig_df.to_markdown(index=False))

print("\n" + "=" * 80)
# Adult UCI significance
print("  SIGNIFICANCE TESTING (Paired t-test): XGBoost (Tuned) vs XGBoost (Tuned + RW)")
print("  Dataset: Adult UCI")
print("=" * 80)
adult_sig_df = perform_paired_ttest(adult_metrics, models[0], models[1], metrics_list)
print(adult_sig_df.to_markdown(index=False))

# Save results
rec_sig_df["Dataset"] = "Recruitment"
adult_sig_df["Dataset"] = "Adult UCI"
combined_sig = pd.concat([rec_sig_df, adult_sig_df], ignore_index=True)
combined_sig.to_csv("outputs/significance_testing.csv", index=False)
print("\n[SAVED] outputs/significance_testing.csv")
print("=" * 80 + "\n")
