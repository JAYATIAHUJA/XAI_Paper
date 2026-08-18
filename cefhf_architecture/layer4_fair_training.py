"""
layer4_fair_training.py -- Layer 4: Fairness-constrained prediction -> h*.

Implements the three options the review lists for Layer 4 (pick 1-2, name them
precisely, drop the undefined "FairXGBoost"):
  1. Counterfactual-augmentation training (CEFHF-specific; uses Layer-2's SCM).
     For every training row generate its SCM counterfactual twin (flip A,
     propagate descendants), keep the SAME label, and train XGBoost on
     D union D_cf with the twins weighted by lambda.  Higher lambda -> stronger
     pressure for p(x) ~= p(x_cf) -> lower CFVR-SCM.  This directly targets the
     counterfactual metric, which is why it is the CEFHF choice.
  2. Fairlearn reductions (ExponentiatedGradient) with an XGBoost base learner
     and a Demographic-Parity / Equalized-Odds constraint -- the standard,
     citable statistical-fair comparator.
  3. FairGBM (Cruz et al. ICLR 2023) -- attempted; documented if unavailable.

The lambda-sweep produces the accuracy-vs-CFVR-SCM Pareto curve, which is the
empirical test of H2 (replaces Proposition 2).

Outputs: outputs/layer4_lambda_sweep.csv, outputs/lambda_sweep_pareto.png,
         outputs/layer4_fairlearn.csv, data/layer4_models.pkl

Run:  python layer4_fair_training.py
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

LAMBDAS = common.CFG["lambda_sweep"]
FAIRLEARN_EPS = [0.01, 0.05, 0.10, 0.20]
DATASETS = ["synthetic", "adult"]


def _split(bundle, seed=42):
    df = bundle.df
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    return df.iloc[idx[:cut]], df.iloc[idx[cut:]]


def _proba(model, X):
    """predict_proba that also handles fairlearn ExponentiatedGradient (which
    stores an ensemble of weighted base predictors in `_predictors`)."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    preds = getattr(model, "_predictors", None)
    if preds:
        ws = getattr(model, "_weights", None)
        ws = np.asarray(ws) if ws is not None else np.ones(len(preds))
        ws = ws / ws.sum()
        return sum(w * p.predict_proba(X)[:, 1] for w, p in zip(ws, preds))
    return model.predict(X).astype(float)


def _eval(model, bundle, test_df, scm):
    X = bundle.transform(test_df)
    y = test_df[bundle.target].to_numpy()
    A = test_df[bundle.sensitive_enc_col].to_numpy()
    p = _proba(model, X)
    pred = (p >= 0.5).astype(int)
    perf = common.perf_metrics(y, pred, p)
    sf = common.stat_fairness(y, pred, A)
    # CFVR-SCM
    cf = scm.flip_protected(test_df, bundle.sensitive)
    pcf = _proba(model, bundle.transform(cf))
    cfvr = float((np.abs(pcf - p) > common.CFG["cfvr"]["tau_prob"]).mean())
    return {**perf, **{k.replace("|", ""): v for k, v in sf.items()}, "CFVR-SCM": cfvr}


# --------------------------------------------------------------------------- #
# 1. Counterfactual-augmentation training
# --------------------------------------------------------------------------- #
def train_cf_aug(bundle, train_df, scm, lam, seed=42):
    X = bundle.transform(train_df)
    y = train_df[bundle.target].to_numpy()
    if lam == 0:
        m = common.make_xgb(seed)
        m.fit(X, y)
        return m
    cf = scm.flip_protected(train_df, bundle.sensitive)
    Xcf = bundle.transform(cf)
    Xall = np.vstack([X, Xcf])
    yall = np.concatenate([y, y])
    wall = np.concatenate([np.ones(len(y)), lam * np.ones(len(y))])
    m = common.make_xgb(seed)
    m.fit(Xall, yall, sample_weight=wall)
    return m


def lambda_sweep(bundle, train_df, test_df, scm):
    rows = []
    models = {}
    for lam in LAMBDAS:
        m = train_cf_aug(bundle, train_df, scm, lam)
        ev = _eval(m, bundle, test_df, scm)
        ev.update({"dataset": bundle.name, "method": "CF-augmentation", "lambda": lam})
        rows.append(ev)
        models[f"cfaug_lam={lam}"] = m
        print(f"    lam={lam:>4} AUC={ev['ROC-AUC']:.3f} CFVR-SCM={ev['CFVR-SCM']:.3f} "
              f"SPD={ev['SPD']:.3f}")
    return rows, models


# --------------------------------------------------------------------------- #
# 2. Fairlearn reductions (ExponentiatedGradient)
# --------------------------------------------------------------------------- #
def fairlearn_sweep(bundle, train_df, test_df, scm):
    try:
        from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    except Exception as e:
        print(f"    [skip] fairlearn reductions unavailable: {e}")
        return [], {}
    X = bundle.transform(train_df)
    y = train_df[bundle.target].to_numpy()
    A = train_df[bundle.sensitive_enc_col].to_numpy()
    rows = []; models = {}
    for eps in FAIRLEARN_EPS:
        try:
            base = common.make_xgb(common.CFG["seeds"]["model_init"], n_estimators=100, max_depth=4)
            eg = ExponentiatedGradient(base, constraints=DemographicParity(difference_bound=eps),
                                       eps=eps, max_iter=20, nu=1e-3)
            eg.fit(X, y, sensitive_features=A)
            ev = _eval(eg, bundle, test_df, scm)  # EG exposes predict_proba via ensemble? guard below
            ev.update({"dataset": bundle.name, "method": "fairlearn-EG-DP", "lambda": eps})
            rows.append(ev); models[f"eg_eps={eps}"] = eg
            print(f"    EG eps={eps:>4} AUC={ev['ROC-AUC']:.3f} CFVR-SCM={ev['CFVR-SCM']:.3f} SPD={ev['SPD']:.3f}")
        except Exception as e:
            print(f"    EG eps={eps} failed: {e}")
    return rows, models


# --------------------------------------------------------------------------- #
# 3. FairGBM (best-effort)
# --------------------------------------------------------------------------- #
def fairgbm_attempt(bundle, train_df, test_df, scm):
    try:
        from fairgbm import FairGBMClassifier  # noqa
        return {"note": "FairGBM import OK (full integration deferred)", "ran": False}
    except Exception as e:
        return {"note": f"FairGBM unavailable on this stack ({e.__class__.__name__})", "ran": False}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 70)
    print("  LAYER 4 -- Fairness-constrained prediction (lambda-sweep -> Pareto)")
    print("=" * 70)
    all_rows = []; all_models = {}
    for name in DATASETS:
        print(f"\n--- {name} ---")
        bundle = common.load_dataset(name)
        _, scm = L2.get_scm(name)
        train_df, test_df = _split(bundle)
        if len(test_df) > 3000:
            test_df = test_df.sample(3000, random_state=42)

        print("  [CF-augmentation lambda-sweep]")
        r1, m1 = lambda_sweep(bundle, train_df, test_df, scm)
        all_rows += r1; all_models.update({f"{name}/{k}": v for k, v in m1.items()})

        print("  [fairlearn ExponentiatedGradient]")
        r2, m2 = fairlearn_sweep(bundle, train_df, test_df, scm)
        all_rows += r2; all_models.update({f"{name}/{k}": v for k, v in m2.items()})

        fb = fairgbm_attempt(bundle, train_df, test_df, scm)
        print(f"  [FairGBM] {fb['note']}")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(common.OUT_DIR, "layer4_results.csv"), index=False)
    print("\n[SAVED] outputs/layer4_results.csv")
    common.save_pickle(all_models, "layer4_models.pkl")
    print("[SAVED] data/layer4_models.pkl")

    # Pareto plot: AUC vs CFVR-SCM for the CF-augmentation lambda-sweep (H2)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 5))
    for i, name in enumerate(DATASETS):
        sub = df_all[(df_all.dataset == name) & (df_all.method == "CF-augmentation")].sort_values("lambda")
        ax = axes[i]
        ax.plot(sub["CFVR-SCM"], sub["ROC-AUC"], "o-", color="#C44E52", label="CF-augmentation (lambda sweep)")
        for _, r in sub.iterrows():
            ax.annotate(f"λ={r['lambda']}", (r["CFVR-SCM"], r["ROC-AUC"]), fontsize=7,
                        textcoords="offset points", xytext=(4, -8))
        # unconstrained reference (lambda=0 already in sweep)
        ax.set_xlabel("CFVR-SCM (lower = fairer)")
        if i == 0: ax.set_ylabel("ROC-AUC")
        ax.set_title(f"{name}: accuracy-fairness Pareto (H2)")
        ax.legend(fontsize=8)
    fig.suptitle("Layer 4: lambda-sweep Pareto frontier (replaces Proposition 2 / H2)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(common.OUT_DIR, "lambda_sweep_pareto.png"), dpi=150)
    plt.close()
    print("[SAVED] outputs/lambda_sweep_pareto.png")
    print("\n[OK] Layer 4 complete.")


if __name__ == "__main__":
    main()
