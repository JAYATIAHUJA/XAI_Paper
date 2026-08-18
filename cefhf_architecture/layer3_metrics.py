"""
layer3_metrics.py -- Layer 3: Fairness evaluation metric stack.

Computes, per model per dataset, with bootstrap CIs:
  * Performance:        Accuracy, F1, ROC-AUC
  * Statistical:        |SPD|, DIR, |EOD|           (fairlearn, via common)
  * CFVR-flip:          flip the protected column only -- the honest
                        direct-effect / unawareness-violation baseline (review P3)
  * CFVR-SCM:           counterfactual via the Layer-2 SCM (abduction-action-
                        prediction) -- the REAL counterfactual fairness metric
  * CFVR-SCM (direct):  synthetic-only -- flip A but freeze the structural
                        mediators so only the direct A->score edge propagates
  * Differential Fairness epsilon across intersectional groups (gender x age)
  * tau-sweep for CFVR-SCM

Recruitment has no SCM (null-control) -> CFVR-SCM = NaN, only CFVR-flip is
reported, exactly as the review prescribes (it should look like noise/instability).

Outputs: outputs/layer3_results.csv, outputs/layer3_results.png

Run:  python layer3_metrics.py
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

DATASETS = ["synthetic", "adult", "recruitment"]
MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "XGBoost+RW"]
TAU_SWEEP = common.CFG["cfvr"]["tau_sweep"]
BOOT_ITERS = common.CFG["bootstrap_iters"]
BOOT_SEED = common.CFG["seeds"]["bootstrap"]


def _cfvr_from_deltas(deltas: np.ndarray, tau: float) -> float:
    return float((np.abs(deltas) > tau).mean())


def _bootstrap_ci_metric(deltas_prob, pred_o, pred_c, y, A_pred, groups, fn, iters, seed):
    """Bootstrap a scalar metric fn(deltas_prob, pred_o, pred_c, y, A_pred, groups)."""
    rng = np.random.default_rng(seed)
    n = len(deltas_prob)
    vals = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(deltas_prob[idx], pred_o[idx], pred_c[idx], y[idx],
                           A_pred[idx], groups[idx]))
        except Exception:
            continue
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(vals)
    return float(np.mean(arr)), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def evaluate_model(model, bundle, test_df, scm=None, freeze_mediators=None):
    """Train already fitted on train. Evaluate on test_df (raw rows)."""
    X_test = bundle.transform(test_df)
    y_test = test_df[bundle.target].to_numpy()
    A_test = test_df[bundle.sensitive_enc_col].to_numpy()
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)

    out = {}
    out.update(common.perf_metrics(y_test, pred, p))
    out.update(common.stat_fairness(y_test, pred, A_test))

    # CFVR-flip (direct-effect baseline)
    cfvr_flip_prob, cfvr_flip_class = common.compute_cfvr_flip(
        model, X_test, bundle.sensitive_idx, tau=common.CFG["cfvr"]["tau_prob"])
    out["CFVR-flip"] = cfvr_flip_prob
    out["CFVR-flip-class"] = cfvr_flip_class

    # CFVR-SCM (real counterfactual via SCM)
    if scm is not None:
        cf_raw = scm.flip_protected(test_df, bundle.sensitive)
        X_cf = bundle.transform(cf_raw)
        p_cf = model.predict_proba(X_cf)[:, 1]
        deltas = p_cf - p
        out["CFVR-SCM"] = _cfvr_from_deltas(deltas, common.CFG["cfvr"]["tau_prob"])
        out["CFVR-SCM-class"] = float((pred != (p_cf >= 0.5).astype(int)).mean())
        # direct variant: flip A but freeze structural mediators (synthetic only)
        if freeze_mediators is not None:
            interv = {bundle.sensitive: cf_raw[bundle.sensitive].to_numpy()}
            for m in freeze_mediators:
                interv[m] = test_df[m].to_numpy()      # freeze at observed
            cf_d = scm.counterfactual(test_df, interv)
            p_cd = model.predict_proba(bundle.transform(cf_d))[:, 1]
            out["CFVR-SCM-direct"] = _cfvr_from_deltas(p_cd - p, common.CFG["cfvr"]["tau_prob"])
        else:
            out["CFVR-SCM-direct"] = float("nan")
        # tau-sweep (use precomputed deltas)
        out["tau_sweep"] = {t: _cfvr_from_deltas(deltas, t) for t in TAU_SWEEP}
    else:
        out["CFVR-SCM"] = float("nan")
        out["CFVR-SCM-class"] = float("nan")
        out["CFVR-SCM-direct"] = float("nan")
        out["tau_sweep"] = {t: float("nan") for t in TAU_SWEEP}

    # Differential Fairness epsilon (intersectional: A x age band)
    if "age" in test_df.columns:
        groups = common.intersectional_groups(A_test, common.age_bands(test_df["age"].to_numpy()))
    else:
        groups = common.intersectional_groups(A_test, A_test)
    out["DF-epsilon"] = common.differential_fairness_epsilon(pred, groups)

    return out


def fit_models(bundle, train_df, seed=42):
    X_train = bundle.transform(train_df)
    y_train = train_df[bundle.target].to_numpy()
    A_train = train_df[bundle.sensitive_enc_col].to_numpy()
    w = common.reweighing_weights(y_train, A_train)
    models = {}
    for name, mk in common.make_baselines(seed).items():
        m = mk()
        m.fit(X_train, y_train)
        models[name] = m
    # reweighed XGBoost
    m = common.make_xgb(seed)
    m.fit(X_train, y_train, sample_weight=w)
    models["XGBoost+RW"] = m
    return models


def main():
    print("=" * 70)
    print("  LAYER 3 -- Fairness evaluation metric stack")
    print("=" * 70)
    rows = []
    for name in DATASETS:
        print(f"\n--- {name} ---")
        bundle = common.load_dataset(name)
        scm = None
        freeze = None
        if name in ("synthetic", "adult"):
            _, scm = L2.get_scm(name)
        if name == "synthetic":
            freeze = ["gap_years", "negotiation"]      # structural mediators of gender
        # 80/20 split
        df = bundle.df
        idx = np.arange(len(df))
        rng = np.random.default_rng(common.CFG["seeds"]["global"])
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        train_df, test_df = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
        # subsample test for SCM counterfactual speed
        if scm is not None and len(test_df) > 3000:
            test_df = test_df.sample(3000, random_state=42)
        models = fit_models(bundle, train_df)
        for mname, model in models.items():
            res = evaluate_model(model, bundle, test_df, scm, freeze)
            res["dataset"] = name; res["model"] = mname
            rows.append(res)
            print(f"  {mname:18s} AUC={res['ROC-AUC']:.3f} |SPD|={res['SPD']:.3f} "
                  f"CFVR-flip={res['CFVR-flip']:.3f} CFVR-SCM={res['CFVR-SCM']:.3f} "
                  f"DFeps={res['DF-epsilon']:.3f}")

    df_res = pd.DataFrame(rows)
    # order columns
    cols = ["dataset", "model", "Accuracy", "F1", "ROC-AUC", "SPD", "DIR", "EOD",
            "CFVR-flip", "CFVR-flip-class", "CFVR-SCM", "CFVR-SCM-class", "CFVR-SCM-direct",
            "DF-epsilon", "tau_sweep"]
    df_res = df_res[[c for c in cols if c in df_res.columns]]
    df_res.to_csv(os.path.join(common.OUT_DIR, "layer3_results.csv"), index=False)
    print("\n[SAVED] outputs/layer3_results.csv")
    # also dump tau-sweeps flat
    sweep_rows = []
    for r in rows:
        for t, v in r["tau_sweep"].items():
            sweep_rows.append({"dataset": r["dataset"], "model": r["model"], "tau": t, "CFVR-SCM": v})
    pd.DataFrame(sweep_rows).to_csv(os.path.join(common.OUT_DIR, "cfvr_tau_sweep.csv"), index=False)
    print("[SAVED] outputs/cfvr_tau_sweep.csv")

    # plot: CFVR-flip vs CFVR-SCM per model per dataset (the headline L3 figure)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 5), sharey=True)
    for i, name in enumerate(DATASETS):
        sub = df_res[df_res.dataset == name]
        x = np.arange(len(sub)); w = 0.4
        axes[i].bar(x - w/2, sub["CFVR-flip"], w, label="CFVR-flip (direct-effect)", color="#8172B3")
        axes[i].bar(x + w/2, sub["CFVR-SCM"], w, label="CFVR-SCM (counterfactual)", color="#C44E52")
        axes[i].set_xticks(x); axes[i].set_xticklabels(sub["model"], rotation=20, ha="right")
        axes[i].set_title(name); axes[i].set_ylim(0, 1)
        if i == 0: axes[i].set_ylabel("violation rate"); axes[i].legend(fontsize=8)
    fig.suptitle("Layer 3: CFVR-flip vs CFVR-SCM (SCM catches mediated bias the flip-test misses)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(common.OUT_DIR, "layer3_cfvr_flip_vs_scm.png"), dpi=150)
    plt.close()
    print("[SAVED] outputs/layer3_cfvr_flip_vs_scm.png")
    print("\n[OK] Layer 3 complete.")


if __name__ == "__main__":
    main()
