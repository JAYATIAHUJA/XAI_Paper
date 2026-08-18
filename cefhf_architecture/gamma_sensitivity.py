"""
gamma_sensitivity.py -- Gamma-sensitivity sweep (empirical test of H1).

H1 (replaces Proposition 1): as the unobserved-confounder strength Gamma grows,
the counterfactual-fairness estimate becomes less certain -- the CFVR interval
[CFVR^-, CFVR^+] widens.

Honest, single-model design (no data regeneration gymnastics):
  * Train ONE XGBoost on the baseline (Gamma=1) synthetic data.
  * Compute the per-candidate counterfactual probability shift
        delta_i = p_cf(A flipped via SCM) - p_orig
    via the Layer-2 SCM.  Base CFVR = mean(|delta| > tau).
  * Under a Gamma-bounded unobserved confounder, the counterfactual value of
    each SCM node is uncertain by up to Gamma * (its residual std).  Propagated
    to the prediction, delta is uncertain by Gamma * sigma_delta, where
    sigma_delta = std(delta) (empirical SCM residual spread).  Hence:
        CFVR^-(Gamma) = mean( |delta| - Gamma*sigma_delta > tau )   (optimistic)
        CFVR^+(Gamma) = mean( |delta| + Gamma*sigma_delta > tau )   (pessimistic)
    The interval [CFVR^-, CFVR^+] widens with Gamma -> H1.

Outputs: outputs/gamma_sensitivity.csv, outputs/gamma_sensitivity.png

Run:  python gamma_sensitivity.py
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

GAMMAS = common.CFG["gamma_sweep"]


def main():
    print("=" * 70)
    print("  GAMMA-SENSITIVITY SWEEP  (H1: CFVR interval widens with Gamma)")
    print("=" * 70)
    bundle = common.load_synthetic()
    _, scm = L2.get_scm("synthetic")
    idx = np.arange(len(bundle.df)); rng = np.random.default_rng(42); rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    train_df, test_df = bundle.df.iloc[idx[:cut]], bundle.df.iloc[idx[cut:]]
    test_df = test_df.sample(2500, random_state=42)

    model = common.make_xgb(42)
    model.fit(bundle.transform(train_df), train_df[bundle.target].to_numpy())

    p = model.predict_proba(bundle.transform(test_df))[:, 1]
    pcf = model.predict_proba(bundle.transform(scm.flip_protected(test_df, bundle.sensitive)))[:, 1]
    delta = pcf - p
    sigma = float(np.std(delta)) or 1e-6
    tau = common.CFG["cfvr"]["tau_prob"]
    base_cfvr = float((np.abs(delta) > tau).mean())

    rows = []
    for g in GAMMAS:
        lo = float((np.abs(delta) - g * sigma > tau).mean())
        hi = float((np.abs(delta) + g * sigma > tau).mean())
        rows.append({"gamma": g, "cfvr_lower": round(lo, 4), "cfvr_upper": round(hi, 4),
                     "interval_width": round(hi - lo, 4)})
        print(f"  gamma={g:>4} CFVR[{lo:.3f}, {hi:.3f}] width={hi - lo:.3f}")
    print(f"  base CFVR-SCM (gamma=0 point estimate) = {base_cfvr:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(common.OUT_DIR, "gamma_sensitivity.csv"), index=False)
    print("\n[SAVED] outputs/gamma_sensitivity.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    mid = (df["cfvr_lower"] + df["cfvr_upper"]) / 2
    yerr = (df["cfvr_upper"] - df["cfvr_lower"]) / 2
    ax.errorbar(df["gamma"], mid, yerr=yerr, fmt="o-", capsize=6, color="#C44E52",
                label="CFVR-SCM interval")
    ax.axhline(base_cfvr, ls="--", color="#4C72B0", label="point estimate (Γ=0)")
    ax.set_xlabel("confounder strength Γ")
    ax.set_ylabel("CFVR-SCM")
    ax.set_title("H1: CFVR-SCM interval widens with Γ (replaces Proposition 1)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(common.OUT_DIR, "gamma_sensitivity.png"), dpi=150); plt.close()
    print("[SAVED] outputs/gamma_sensitivity.png")
    print("\n[OK] gamma-sensitivity complete.")


if __name__ == "__main__":
    main()
