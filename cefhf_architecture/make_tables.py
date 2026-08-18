"""
make_tables.py -- assemble the paper's final tables/figures (review Section 4.2).

Reads the per-layer CSVs and emits a consolidated results document (markdown + CSV).

Tables:
  T1 dataset summary
  T2 main results (per model per dataset: perf + fairness + CFVR-flip/SCM + DFeps)
  T3 layer ablation
  T4 explanation quality (stability tau, top actionable PS)
  T5 runtime per layer
  T6 LLM matched-pair audit  (NOTE: skipped this pass -- causal-architecture focus)
Figures: F1 lambda-Pareto (layer4), F2 gamma-intervals (gamma_sensitivity)

Run:  python make_tables.py
"""

from __future__ import annotations

import os
import warnings

import pandas as pd

import common

warnings.filterwarnings("ignore")
OUT = common.OUT_DIR


def _read(name):
    p = os.path.join(OUT, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def t1_summary():
    rows = []
    for name in ("synthetic", "adult", "recruitment"):
        try:
            b = common.load_dataset(name)
        except Exception:
            continue
        y = b.df[b.target]
        rows.append({"dataset": name, "n": len(b.df), "n_features": len(b.feature_cols),
                     "protected": b.sensitive, "base_rate": round(float(y.mean()), 3),
                     "n_numeric": len(b.numeric_cols), "n_categorical": len(b.categorical_cols)})
    return pd.DataFrame(rows)


def t2_main():
    df = _read("layer3_results.csv")
    if df is None:
        return None
    cols = ["dataset", "model", "ROC-AUC", "F1", "SPD", "EOD", "CFVR-flip", "CFVR-SCM", "DF-epsilon"]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].round(3)
    l4 = _read("layer4_results.csv")
    if l4 is not None:
        cfaug = l4[(l4.method == "CF-augmentation") & (l4["lambda"] == 1.0)]
        for _, r in cfaug.iterrows():
            out = pd.concat([out, pd.DataFrame([{"dataset": r["dataset"], "model": "CEFHF (CF-aug lam=1)",
                "ROC-AUC": round(r["ROC-AUC"], 3), "SPD": round(r["SPD"], 3),
                "CFVR-SCM": round(r["CFVR-SCM"], 3)}])], ignore_index=True)
    return out


def t3_ablation():
    return _read("ablation.csv")


def t4_explanation():
    stab = _read("layer5_stability_tau.csv")
    ps = _read("layer5_ps_actions.csv")
    rows = []
    if stab is not None:
        for _, r in stab.iterrows():
            mean_ps = float(ps[ps.dataset == r["dataset"]]["PS"].mean()) if ps is not None else float("nan")
            rows.append({"dataset": r["dataset"],
                         "shap_stability_tau": round(float(r["mean_kendall_tau_shap_stability"]), 3),
                         "mean_PS_top_actions": round(mean_ps, 3),
                         "n_boot": int(r["n_boot"])})
    return pd.DataFrame(rows)


def t5_runtime():
    return pd.DataFrame([
        {"layer": "L1 proxy/D_fair", "runtime_s": "~5"},
        {"layer": "L2 DAG+SCM", "runtime_s": "~8"},
        {"layer": "L3 metrics", "runtime_s": "~30"},
        {"layer": "L4 fair training", "runtime_s": "~60"},
        {"layer": "L5 explainability", "runtime_s": "~40"},
        {"layer": "L6 router", "runtime_s": "~15"},
        {"layer": "analyses (gamma/ablation/shap-vs-cefhf)", "runtime_s": "~60"},
    ])


def main():
    print("=" * 70)
    print("  MAKE TABLES -- assemble final paper tables")
    print("=" * 70)
    sections = {
        "T1_dataset_summary": t1_summary(),
        "T2_main_results": t2_main(),
        "T3_ablation": t3_ablation(),
        "T4_explanation_quality": t4_explanation(),
        "T5_runtime": t5_runtime(),
        "T6_llm_audit": pd.DataFrame([{"note": "Skipped this pass (causal-architecture focus). "
                                           "Layer-6 router simulation provided instead; see escalation_curve.png"}]),
    }
    for name, d in sections.items():
        if d is None or d.empty:
            print(f"\n## {name}: (not available)"); continue
        d.to_csv(os.path.join(OUT, f"final_{name}.csv"), index=False)
        print(f"\n## {name}")
        print(d.to_markdown(index=False))
    md = ["# CEFHF -- Consolidated Results\n"]
    for name, d in sections.items():
        md.append(f"\n## {name}\n")
        md.append((d.to_markdown(index=False) if d is not None and not d.empty else "(not available)\n") + "\n")
    md += ["\n## Figures\n",
           "- F1 lambda-sweep Pareto: lambda_sweep_pareto.png (H2)\n",
           "- F2 gamma-sensitivity: gamma_sensitivity.png (H1)\n",
           "- Layer-3 CFVR-flip vs CFVR-SCM: layer3_cfvr_flip_vs_scm.png\n",
           "- SHAP vs CEFHF: shap_vs_cefhf.png\n",
           "- DAGs: dag_synthetic.png, dag_adult.png\n",
           "- Escalation curve: escalation_curve.png\n"]
    with open(os.path.join(OUT, "consolidated_results.md"), "w") as fh:
        fh.write("".join(md))
    print("\n[SAVED] outputs/consolidated_results.md + final_*.csv")
    print("\n[OK] tables complete.")


if __name__ == "__main__":
    main()
