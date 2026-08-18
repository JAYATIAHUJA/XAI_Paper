"""
ablation.py -- Layer ablation study (review Section 4.2 T3).

Drops each layer from the full CEFHF pipeline and reports the effect on the
core metrics (AUC, |SPD|, CFVR-SCM), to show each layer's marginal contribution.

  Full CEFHF        = L1 (proxy-removal D_fair) + L2 (SCM) + L3 (metrics)
                       + L4 (CF-augmentation, lambda*) + L5 (actionability filter)
  drop L1           = train on raw features (no proxy removal)
  drop L2/SCM       = CFVR reduces to the flip-test only (no counterfactual engine)
  drop L4           = unconstrained XGBoost (no fairness-constrained training)
  drop L5           = no actionability filter on PS (immutable features stay actionable)

Outputs: outputs/ablation.csv, outputs/ablation.png

Run:  python ablation.py
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import layer2_dag_scm as L2

warnings.filterwarnings("ignore")
DATASETS = ["synthetic", "adult"]
LAMBDA_STAR = 1.0          # chosen from the Layer-4 Pareto (best CFVR-SCM w/o AUC loss)


def _split(bundle, seed=42):
    idx = np.arange(len(bundle.df)); rng = np.random.default_rng(seed); rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    return bundle.df.iloc[idx[:cut]], bundle.df.iloc[idx[cut:]]


def _proba(model, X):
    return model.predict_proba(X)[:, 1]


def cfvr_scm(model, bundle, test_df, scm):
    p = _proba(model, bundle.transform(test_df))
    pcf = _proba(model, bundle.transform(scm.flip_protected(test_df, bundle.sensitive)))
    return float((np.abs(pcf - p) > common.CFG["cfvr"]["tau_prob"]).mean())


def evaluate(name, model, bundle, test_df, scm, use_scm=True):
    X = bundle.transform(test_df); y = test_df[bundle.target].to_numpy()
    A = test_df[bundle.sensitive_enc_col].to_numpy()
    p = _proba(model, X); pred = (p >= 0.5).astype(int)
    perf = common.perf_metrics(y, pred, p)
    sf = common.stat_fairness(y, pred, A)
    if use_scm and scm is not None:
        cfvr = cfvr_scm(model, bundle, test_df, scm)
    else:
        cfvr = float(common.compute_cfvr_flip(model, X, bundle.sensitive_idx)[0])
    return {"AUC": perf["ROC-AUC"], "SPD": sf["SPD"], "CFVR": cfvr}


def run_config(bundle, train_df, test_df, scm, variant):
    """The integrated pipeline = L1 (proxy-removal) + L2 (SCM) + L3 (metrics)
    + L4 (CF-augmentation, lambda*).  Each variant drops one layer."""
    if variant == "Full CEFHF":
        b = _reduced_bundle(bundle)            # L1 proxy-removal applied
        m = _train_cf_aug(b, train_df, scm, LAMBDA_STAR)
        return evaluate(variant, m, b, test_df, scm, use_scm=True)
    if variant == "drop L1":
        m = _train_cf_aug(bundle, train_df, scm, LAMBDA_STAR)   # full features, no L1
        return evaluate(variant, m, bundle, test_df, scm, use_scm=True)
    if variant == "drop L2/SCM":
        m = _train_cf_aug(bundle, train_df, scm, LAMBDA_STAR)   # CF-aug, but metric = flip-test
        return evaluate(variant, m, bundle, test_df, scm, use_scm=False)
    if variant == "drop L4":
        m = common.make_xgb(42); m.fit(bundle.transform(train_df), train_df[bundle.target].to_numpy())
        return evaluate(variant, m, bundle, test_df, scm, use_scm=True)
    if variant == "drop L5":
        m = _train_cf_aug(bundle, train_df, scm, LAMBDA_STAR)
        ev = evaluate(variant, m, bundle, test_df, scm, use_scm=True)
        ev["note"] = "L5 affects explanations, not model metrics; see layer5 outputs"
        return ev
    return {}


def _reduced_bundle(bundle):
    """Rebuild the bundle with L1-flagged proxy features removed (proxy-removal
    D_fair from Layer 1).  The df keeps every column (the SCM still needs all DAG
    nodes); only the *model's* feature set shrinks."""
    try:
        dfair = common.load_pickle(f"dfair_proxyremoval_{bundle.name}.pkl")
        dropped = set(dfair["dropped"])
    except Exception:
        dropped = set()
    keep = [f for f in bundle.feature_cols if f not in dropped]
    return common._finalize(bundle.name, bundle.df, bundle.target, bundle.sensitive, keep, bundle.legitimate)


def _train_cf_aug(bundle, train_df, scm, lam):
    X = bundle.transform(train_df); y = train_df[bundle.target].to_numpy()
    cf = scm.flip_protected(train_df, bundle.sensitive); Xcf = bundle.transform(cf)
    Xall = np.vstack([X, Xcf]); yall = np.concatenate([y, y])
    wall = np.concatenate([np.ones(len(y)), lam * np.ones(len(y))])
    m = common.make_xgb(42); m.fit(Xall, yall, sample_weight=wall)
    return m


def main():
    print("=" * 70)
    print("  ABLATION STUDY")
    print("=" * 70)
    rows = []
    variants = ["Full CEFHF", "drop L1", "drop L2/SCM", "drop L4", "drop L5"]
    for name in DATASETS:
        print(f"\n--- {name} ---")
        bundle = common.load_dataset(name)
        _, scm = L2.get_scm(name)
        train_df, test_df = _split(bundle)
        if len(test_df) > 2500:
            test_df = test_df.sample(2500, random_state=42)
        for v in variants:
            r = run_config(bundle, train_df, test_df, scm, v)
            r["dataset"] = name; r["variant"] = v
            rows.append(r)
            print(f"  {v:14s} AUC={r['AUC']:.3f} SPD={r['SPD']:.3f} CFVR={r['CFVR']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(common.OUT_DIR, "ablation.csv"), index=False)
    print("\n[SAVED] outputs/ablation.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, name in enumerate(DATASETS):
        sub = df[df.dataset == name]
        x = np.arange(len(variants)); w = 0.35
        axes[i].bar(x - w/2, sub["AUC"], w, label="AUC", color="#4C72B0")
        axes[i].bar(x + w/2, sub["CFVR"], w, label="CFVR-SCM", color="#C44E52")
        axes[i].set_xticks(x); axes[i].set_xticklabels(variants, rotation=20, ha="right")
        axes[i].set_title(name); axes[i].legend(fontsize=8); axes[i].set_ylim(0, 1)
    fig.suptitle("Layer ablation: each layer's marginal contribution", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(common.OUT_DIR, "ablation.png"), dpi=150); plt.close()
    print("[SAVED] outputs/ablation.png")
    print("\n[OK] ablation complete.")


if __name__ == "__main__":
    main()
